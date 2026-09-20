use axis::prelude::*;

pub struct SudokuAxes {
    pub batch: Axis,
    pub position: Axis,
    pub token: Axis,
    pub feature: Axis,
    pub hidden: Axis,
    pub class: Axis,
    pub head: Dim,
    pub head_feature: Dim,
}

impl SudokuAxes {
    pub fn new(embedding: usize, heads: usize) -> Result<Self> {
        if embedding == 0 || heads == 0 || !embedding.is_multiple_of(heads) {
            return Err("embedding must be positive and divisible by heads".into());
        }
        let feature = Axis::new("feature");
        Ok(Self {
            batch: Axis::new("batch"),
            position: Axis::new("position"),
            token: Axis::new("token"),
            feature,
            hidden: Axis::new("feed_forward"),
            class: Axis::new("digit"),
            head: Axis::new("head").of(heads),
            head_feature: Axis::new("head_feature").of(embedding / heads),
        })
    }

    pub fn embedding(&self) -> usize {
        self.head.extent * self.head_feature.extent
    }
}

struct BidirectionalAttention {
    time: Axis,
    feature: Axis,
    head: Dim,
    head_feature: Dim,
    query_time: Axis,
    key_time: Axis,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
}

impl BidirectionalAttention {
    fn new(axes: &SudokuAxes) -> Self {
        let projection = || Linear::new(axes.feature, axes.feature.of(axes.embedding()));
        Self {
            time: axes.position,
            feature: axes.feature,
            head: axes.head,
            head_feature: axes.head_feature,
            query_time: axes.position.role("query_position"),
            key_time: axes.position.role("key_position"),
            query: projection(),
            key: projection(),
            value: projection(),
            output: projection(),
        }
    }

    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let split = |x: &Tensor, time| {
            x.split(self.feature, [self.head, self.head_feature])?
                .rename(self.time, time)
        };
        let q = split(q, self.query_time)?;
        let k = split(k, self.key_time)?;
        let v = split(v, self.key_time)?;
        q.contract(&k, self.head_feature.axis)?
            .scale(1.0 / (self.head_feature.extent as f32).sqrt())?
            .softmax(self.key_time)?
            .contract(&v, self.key_time)?
            .merge([self.head.axis, self.head_feature.axis], self.feature)?
            .rename(self.query_time, self.time)
    }
}

impl Module for BidirectionalAttention {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        input.extent(self.time)?;
        self.output.output_shape(&self.query.output_shape(input)?)
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let output = self.output_shape(input)?;
        let projected = self.query.build(input, device, seed)?;
        self.key.build(input, device, seed.wrapping_add(1))?;
        self.value.build(input, device, seed.wrapping_add(2))?;
        self.output
            .build(&projected, device, seed.wrapping_add(3))?;
        Ok(output)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output.forward(&self.attend(
            &self.query.forward(input)?,
            &self.key.forward(input)?,
            &self.value.forward(input)?,
        )?)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        [
            ("query", &self.query),
            ("key", &self.key),
            ("value", &self.value),
            ("output", &self.output),
        ]
        .into_iter()
        .flat_map(|(prefix, layer)| {
            layer
                .named_parameters()
                .into_iter()
                .map(move |(name, parameter)| (format!("{prefix}.{name}"), parameter))
        })
        .collect()
    }
}

struct TransformerBlock {
    norm1: LayerNorm,
    attention: BidirectionalAttention,
    norm2: LayerNorm,
    feed_forward: Sequential,
}

impl TransformerBlock {
    fn new(axes: &SudokuAxes) -> Result<Self> {
        Ok(Self {
            norm1: LayerNorm::new(axes.feature, 1e-5)?,
            attention: BidirectionalAttention::new(axes),
            norm2: LayerNorm::new(axes.feature, 1e-5)?,
            feed_forward: Sequential::new((
                Linear::new(axes.feature, axes.hidden.of(4 * axes.embedding())),
                GELU,
                Linear::new(axes.hidden, axes.feature.of(axes.embedding())),
            )),
        })
    }
}

impl Module for TransformerBlock {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        let attention = self
            .attention
            .output_shape(&self.norm1.output_shape(input)?)?;
        if &attention != input {
            return Err("attention residual changed shape".into());
        }
        let feed_forward = self
            .feed_forward
            .output_shape(&self.norm2.output_shape(input)?)?;
        if &feed_forward != input {
            return Err("feed-forward residual changed shape".into());
        }
        Ok(input.clone())
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        self.output_shape(input)?;
        let normalized = self.norm1.build(input, device, seed)?;
        self.attention
            .build(&normalized, device, seed.wrapping_add(1))?;
        let normalized = self.norm2.build(input, device, seed.wrapping_add(2))?;
        self.feed_forward
            .build(&normalized, device, seed.wrapping_add(3))?;
        Ok(input.clone())
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let attended = input.add(&self.attention.forward(&self.norm1.forward(input)?)?)?;
        attended.add(&self.feed_forward.forward(&self.norm2.forward(&attended)?)?)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        [
            ("norm1", &self.norm1 as &dyn Module),
            ("attention", &self.attention),
            ("norm2", &self.norm2),
            ("feed_forward", &self.feed_forward),
        ]
        .into_iter()
        .flat_map(|(prefix, module)| {
            module
                .named_parameters()
                .into_iter()
                .map(move |(name, parameter)| (format!("{prefix}.{name}"), parameter))
        })
        .collect()
    }
}

pub struct SudokuTransformer {
    axes: SudokuAxes,
    token_embedding: Linear,
    position_embedding: Option<Parameter>,
    blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    output: Linear,
}

impl SudokuTransformer {
    pub fn new(embedding: usize, heads: usize, layers: usize) -> Result<Self> {
        if layers == 0 {
            return Err("transformer must contain at least one layer".into());
        }
        let axes = SudokuAxes::new(embedding, heads)?;
        let blocks = (0..layers)
            .map(|_| TransformerBlock::new(&axes))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            token_embedding: Linear::new(axes.token, axes.feature.of(axes.embedding())),
            position_embedding: None,
            blocks,
            final_norm: LayerNorm::new(axes.feature, 1e-5)?,
            output: Linear::new(axes.feature, axes.class.of(9)),
            axes,
        })
    }

    pub fn axes(&self) -> &SudokuAxes {
        &self.axes
    }
}

impl Module for SudokuTransformer {
    fn output_shape(&self, input: &Shape) -> Result<Shape> {
        if input.extent(self.axes.position)? != 81 || input.extent(self.axes.token)? != 10 {
            return Err("Sudoku input must have 81 positions and a 10-token one-hot axis".into());
        }
        let mut shape = self.token_embedding.output_shape(input)?;
        for block in &self.blocks {
            shape = block.output_shape(&shape)?;
        }
        self.output
            .output_shape(&self.final_norm.output_shape(&shape)?)
    }

    fn build(&mut self, input: &Shape, device: &Device, seed: u64) -> Result<Shape> {
        let output = self.output_shape(input)?;
        let mut shape = self.token_embedding.build(input, device, seed)?;
        if self.position_embedding.is_none() {
            let mut state = seed.wrapping_add(100).max(1);
            let values = (0..81 * self.axes.embedding())
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (((state >> 40) as f32 / (1_u32 << 24) as f32) * 2.0 - 1.0) * 0.02
                })
                .collect::<Vec<_>>();
            self.position_embedding = Some(Parameter::new(Tensor::from_slice(
                &values,
                [
                    self.axes.position.of(81),
                    self.axes.feature.of(self.axes.embedding()),
                ],
                device,
            )?));
        }
        for (index, block) in self.blocks.iter_mut().enumerate() {
            shape = block.build(&shape, device, seed.wrapping_add(10 + index as u64))?;
        }
        shape = self
            .final_norm
            .build(&shape, device, seed.wrapping_add(200))?;
        self.output.build(&shape, device, seed.wrapping_add(201))?;
        Ok(output)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.output_shape(input.shape())?;
        let mut value = self.token_embedding.forward(input)?.add(
            &self
                .position_embedding
                .as_ref()
                .ok_or("SudokuTransformer must be built before forward")?
                .tensor(),
        )?;
        for block in &self.blocks {
            value = block.forward(&value)?;
        }
        self.output.forward(&self.final_norm.forward(&value)?)
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut parameters = self
            .token_embedding
            .named_parameters()
            .into_iter()
            .map(|(name, parameter)| (format!("token_embedding.{name}"), parameter))
            .collect::<Vec<_>>();
        if let Some(position) = &self.position_embedding {
            parameters.push(("position_embedding".into(), position.clone()));
        }
        for (index, block) in self.blocks.iter().enumerate() {
            parameters.extend(
                block
                    .named_parameters()
                    .into_iter()
                    .map(|(name, parameter)| (format!("blocks.{index}.{name}"), parameter)),
            );
        }
        parameters.extend(
            self.final_norm
                .named_parameters()
                .into_iter()
                .map(|(name, parameter)| (format!("final_norm.{name}"), parameter)),
        );
        parameters.extend(
            self.output
                .named_parameters()
                .into_iter()
                .map(|(name, parameter)| (format!("output.{name}"), parameter)),
        );
        parameters
    }
}
