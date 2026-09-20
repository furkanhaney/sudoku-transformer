//! Generated-data Sudoku transformer in Rust on Axis.
mod model;

use axis::prelude::*;
use model::{SudokuAxes, SudokuTransformer};
use std::{env, time::Instant};

const TRAIN_SEED: u64 = 0x5_d0c0_7a11;
const EVAL_SEED: u64 = 0x000e_7a15_d0c0;
const AUDIT_SEED: u64 = 0xa0d1_700d_5eed;

#[derive(Clone)]
struct SudokuSample {
    puzzle: [u8; 81],
    solution: [u8; 81],
}

struct SudokuGenerator {
    seed: u64,
    next_id: u64,
    blanks: usize,
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn shuffle<T>(&mut self, values: &mut [T]) {
        for end in (1..values.len()).rev() {
            values.swap(end, self.next() as usize % (end + 1));
        }
    }
}

impl SudokuGenerator {
    fn new(seed: u64, blanks: usize) -> Result<Self> {
        if !(1..81).contains(&blanks) {
            return Err("Sudoku blank count must be within 1..81".into());
        }
        Ok(Self {
            seed,
            next_id: 0,
            blanks,
        })
    }

    fn generate(&self, id: u64) -> SudokuSample {
        let mut rng = Rng::new(self.seed ^ id.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let mut digits = [1_u8, 2, 3, 4, 5, 6, 7, 8, 9];
        rng.shuffle(&mut digits);
        let mut bands = [0, 1, 2];
        let mut stacks = [0, 1, 2];
        rng.shuffle(&mut bands);
        rng.shuffle(&mut stacks);
        let mut rows = [0; 9];
        let mut columns = [0; 9];
        for (out_band, band) in bands.into_iter().enumerate() {
            let mut within = [0, 1, 2];
            rng.shuffle(&mut within);
            for (offset, row) in within.into_iter().enumerate() {
                rows[out_band * 3 + offset] = band * 3 + row;
            }
        }
        for (out_stack, stack) in stacks.into_iter().enumerate() {
            let mut within = [0, 1, 2];
            rng.shuffle(&mut within);
            for (offset, column) in within.into_iter().enumerate() {
                columns[out_stack * 3 + offset] = stack * 3 + column;
            }
        }
        let mut solution = [0_u8; 81];
        for output_row in 0..9 {
            for output_column in 0..9 {
                let row = rows[output_row];
                let column = columns[output_column];
                let canonical = (row * 3 + row / 3 + column) % 9;
                solution[output_row * 9 + output_column] = digits[canonical];
            }
        }
        let mut positions = std::array::from_fn::<_, 81, _>(|index| index);
        rng.shuffle(&mut positions);
        let mut puzzle = solution;
        for &position in &positions[..self.blanks] {
            puzzle[position] = 0;
        }
        SudokuSample { puzzle, solution }
    }
}

impl DataSource for SudokuGenerator {
    type Item = SudokuSample;

    fn next_sample(&mut self) -> Result<Option<Sample<Self::Item>>> {
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or("Sudoku generator exhausted its sample identity space")?;
        Ok(Some(Sample {
            id: (u128::from(self.seed) << 64) | u128::from(id),
            value: self.generate(id),
        }))
    }

    fn available(&self) -> Option<usize> {
        None
    }
}

fn tensors(
    samples: &[SudokuSample],
    axes: &SudokuAxes,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut tokens = vec![0.0; samples.len() * 81 * 10];
    let mut targets = vec![0.0; samples.len() * 81 * 9];
    let mut blanks = vec![0.0; samples.len() * 81];
    for (batch, sample) in samples.iter().enumerate() {
        for position in 0..81 {
            let puzzle = usize::from(sample.puzzle[position]);
            let solution = usize::from(sample.solution[position] - 1);
            tokens[(batch * 81 + position) * 10 + puzzle] = 1.0;
            targets[(batch * 81 + position) * 9 + solution] = 1.0;
            blanks[batch * 81 + position] = f32::from(puzzle == 0);
        }
    }
    Ok((
        Tensor::from_slice(
            &tokens,
            [
                axes.batch.of(samples.len()),
                axes.position.of(81),
                axes.token.of(10),
            ],
            device,
        )?,
        Tensor::from_slice(
            &targets,
            [
                axes.batch.of(samples.len()),
                axes.position.of(81),
                axes.class.of(9),
            ],
            device,
        )?,
        Tensor::from_slice(
            &blanks,
            [axes.batch.of(samples.len()), axes.position.of(81)],
            device,
        )?,
    ))
}

fn masked_loss(
    logits: &Tensor,
    targets: &Tensor,
    blanks: &Tensor,
    axes: &SudokuAxes,
) -> Result<Tensor> {
    logits
        .categorical_cross_entropy_with_logits(targets, axes.class)?
        .masked_mean(blanks)
}

fn metrics(
    model: &SudokuTransformer,
    samples: &[SudokuSample],
    axes: &SudokuAxes,
    device: &Device,
) -> Result<(f32, f32, f32)> {
    let (inputs, targets, blanks) = tensors(samples, axes, device)?;
    let logits_tensor = model.forward(&inputs)?;
    let loss = masked_loss(&logits_tensor, &targets, &blanks, axes)?.item()?;
    let logits = logits_tensor.to_vec()?;
    let mut correct = 0_usize;
    let mut blanks_total = 0_usize;
    let mut solved = 0_usize;
    for (batch, sample) in samples.iter().enumerate() {
        let mut complete = true;
        for position in 0..81 {
            if sample.puzzle[position] != 0 {
                continue;
            }
            blanks_total += 1;
            let start = (batch * 81 + position) * 9;
            let predicted = logits[start..start + 9]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index + 1)
                .unwrap();
            let matches = predicted == usize::from(sample.solution[position]);
            correct += usize::from(matches);
            complete &= matches;
        }
        solved += usize::from(complete);
    }
    Ok((
        loss,
        correct as f32 / blanks_total as f32,
        solved as f32 / samples.len() as f32,
    ))
}

fn metrics_chunked(
    model: &SudokuTransformer,
    samples: &[SudokuSample],
    chunk_size: usize,
    axes: &SudokuAxes,
    device: &Device,
) -> Result<(f32, f32, f32)> {
    if samples.is_empty() || chunk_size == 0 {
        return Err("evaluation samples and chunk size must be positive".into());
    }
    let mut weighted = (0.0_f64, 0.0_f64, 0.0_f64);
    for chunk in samples.chunks(chunk_size) {
        let measured = metrics(model, chunk, axes, device)?;
        let weight = chunk.len() as f64;
        weighted.0 += f64::from(measured.0) * weight;
        weighted.1 += f64::from(measured.1) * weight;
        weighted.2 += f64::from(measured.2) * weight;
    }
    let total = samples.len() as f64;
    Ok((
        (weighted.0 / total) as f32,
        (weighted.1 / total) as f32,
        (weighted.2 / total) as f32,
    ))
}

fn main() -> Result<()> {
    let mut steps = 100_usize;
    let mut batch_size = 4_usize;
    let mut eval_size = 8_usize;
    let mut eval_batch_size = None;
    let mut eval_every = None;
    let mut audit_size = None;
    let mut log_every = 1_usize;
    let mut blanks = 36_usize;
    let mut embedding = 24_usize;
    let mut heads = 4_usize;
    let mut layers = 2_usize;
    let mut learning_rate = 1e-3_f32;
    let mut weight_decay = 0.01_f32;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--smoke" => {
                steps = 3;
                batch_size = 2;
                eval_size = 2;
                embedding = 12;
                heads = 3;
                layers = 1;
                learning_rate = 3e-3;
            }
            "--steps" => steps = args.next().ok_or("--steps needs a value")?.parse()?,
            "--batch" => batch_size = args.next().ok_or("--batch needs a value")?.parse()?,
            "--blanks" => blanks = args.next().ok_or("--blanks needs a value")?.parse()?,
            "--embedding" => embedding = args.next().ok_or("--embedding needs a value")?.parse()?,
            "--heads" => heads = args.next().ok_or("--heads needs a value")?.parse()?,
            "--layers" => layers = args.next().ok_or("--layers needs a value")?.parse()?,
            "--eval-size" => eval_size = args.next().ok_or("--eval-size needs a value")?.parse()?,
            "--eval-batch" => {
                eval_batch_size = Some(args.next().ok_or("--eval-batch needs a value")?.parse()?)
            }
            "--eval-every" => {
                eval_every = Some(args.next().ok_or("--eval-every needs a value")?.parse()?)
            }
            "--audit-size" => {
                audit_size = Some(args.next().ok_or("--audit-size needs a value")?.parse()?)
            }
            "--log-every" => log_every = args.next().ok_or("--log-every needs a value")?.parse()?,
            "--learning-rate" => {
                learning_rate = args
                    .next()
                    .ok_or("--learning-rate needs a value")?
                    .parse()?
            }
            "--weight-decay" => {
                weight_decay = args.next().ok_or("--weight-decay needs a value")?.parse()?
            }
            "--help" | "-h" => {
                println!(
                    r#"sudoku-transformer [--smoke] [--steps N] [--batch N]
                     [--eval-size N] [--eval-batch N] [--eval-every N]
                     [--audit-size N] [--log-every N]
                     [--blanks N] [--embedding N] [--heads N] [--layers N]
                     [--learning-rate F] [--weight-decay F]"#
                );
                return Ok(());
            }
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    if steps == 0 || batch_size == 0 || eval_size == 0 {
        return Err("steps, batch, and evaluation size must be positive".into());
    }
    let eval_batch_size = eval_batch_size.unwrap_or(batch_size);
    if eval_batch_size == 0 || eval_every == Some(0) || log_every == 0 {
        return Err(
            "evaluation batch, evaluation interval, and log interval must be positive".into(),
        );
    }
    if audit_size == Some(0) {
        return Err("audit size must be positive when requested".into());
    }

    let device = Device::cuda(0)?;
    let mut model = SudokuTransformer::new(embedding, heads, layers)?;
    let input_shape = Shape::new([
        model.axes().batch.of(batch_size),
        model.axes().position.of(81),
        model.axes().token.of(10),
    ])?;
    model.build(&input_shape, &device, 42)?;
    let parameters: usize = model
        .parameters()
        .iter()
        .map(|parameter| parameter.tensor().shape().len())
        .sum();
    println!(
        "model bidirectional-transformer layers={layers} embedding={embedding} heads={heads} parameters={parameters}"
    );

    let mut train = DataLoader::new(SudokuGenerator::new(TRAIN_SEED, blanks)?, batch_size)?
        .assert_idr(IdrLimits::generated(0.0)?)?;
    let mut evaluation = DataLoader::new(SudokuGenerator::new(EVAL_SEED, blanks)?, eval_size)?
        .assert_idr(IdrLimits::generated(0.0)?)?;
    let evaluation = evaluation
        .next_batch()?
        .expect("generated evaluation batch");
    let mut disjoint = TrainEvalDisjoint::new();
    disjoint.observe_evaluation(evaluation.sample_ids.iter().copied())?;
    let audit = if let Some(size) = audit_size {
        let mut source = DataLoader::new(SudokuGenerator::new(AUDIT_SEED, blanks)?, size)?
            .assert_idr(IdrLimits::generated(0.0)?)?;
        let batch = source.next_batch()?.expect("generated audit batch");
        disjoint.observe_evaluation(batch.sample_ids.iter().copied())?;
        let mut tuning_audit_disjoint = TrainEvalDisjoint::new();
        tuning_audit_disjoint.observe_train(evaluation.sample_ids.iter().copied())?;
        tuning_audit_disjoint.observe_evaluation(batch.sample_ids.iter().copied())?;
        Some((batch, tuning_audit_disjoint))
    } else {
        None
    };
    let initial = metrics_chunked(
        &model,
        &evaluation.samples,
        eval_batch_size,
        model.axes(),
        &device,
    )?;
    println!(
        "initial eval_loss={:.6} blank_accuracy={:.2}% solved={:.2}% eval_samples={} eval_batch={}",
        initial.0,
        initial.1 * 100.0,
        initial.2 * 100.0,
        evaluation.samples.len(),
        eval_batch_size,
    );

    let mut trainer = Trainer::new(AdamW::new(learning_rate, weight_decay)?);
    let started = Instant::now();
    let mut last_receipt = None;
    for _ in 0..steps {
        let batch = train.next_batch()?.expect("generated training batch");
        disjoint.observe_train(batch.sample_ids.iter().copied())?;
        let (inputs, targets, blanks) = tensors(&batch.samples, model.axes(), &device)?;
        let report = trainer.step(&mut model, |model| {
            let logits = model.forward(&inputs)?;
            masked_loss(&logits, &targets, &blanks, model.axes())
        })?;
        if report.step() % log_every == 0 || report.step() == 1 || report.step() == steps {
            println!(
                "step={} samples={} pre_update_loss={:.6}",
                report.step(),
                train.samples_delivered(),
                report.pre_update_loss()?
            );
        }
        if eval_every.is_some_and(|interval| report.step() % interval == 0) && report.step() < steps
        {
            let checkpoint = metrics_chunked(
                &model,
                &evaluation.samples,
                eval_batch_size,
                model.axes(),
                &device,
            )?;
            println!(
                "evaluation step={} eval_loss={:.6} blank_accuracy={:.2}% solved={:.2}%",
                report.step(),
                checkpoint.0,
                checkpoint.1 * 100.0,
                checkpoint.2 * 100.0
            );
        }
        last_receipt = Some(batch.regime);
    }
    let final_metrics = metrics_chunked(
        &model,
        &evaluation.samples,
        eval_batch_size,
        model.axes(),
        &device,
    )?;
    println!(
        "final eval_loss={:.6} blank_accuracy={:.2}% solved={:.2}% elapsed_s={:.2}",
        final_metrics.0,
        final_metrics.1 * 100.0,
        final_metrics.2 * 100.0,
        started.elapsed().as_secs_f64()
    );
    if let Some((audit, tuning_audit_disjoint)) = audit {
        let audit_metrics = metrics_chunked(
            &model,
            &audit.samples,
            eval_batch_size,
            model.axes(),
            &device,
        )?;
        println!(
            "final audit_loss={:.6} blank_accuracy={:.2}% solved={:.2}% audit_samples={}",
            audit_metrics.0,
            audit_metrics.1 * 100.0,
            audit_metrics.2 * 100.0,
            audit.samples.len()
        );
        println!("TUNING/AUDIT {}", tuning_audit_disjoint.receipt());
    }
    println!("{}", last_receipt.expect("positive steps"));
    println!("{}", disjoint.receipt());
    if !final_metrics.0.is_finite() {
        return Err("Sudoku acceptance produced a non-finite evaluation loss".into());
    }
    println!("PASS: bidirectional Sudoku transformer trained on fresh generated boards");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid(solution: &[u8; 81]) -> bool {
        for group in 0..9 {
            let mut row = [false; 10];
            let mut column = [false; 10];
            let mut block = [false; 10];
            for offset in 0..9 {
                row[usize::from(solution[group * 9 + offset])] = true;
                column[usize::from(solution[offset * 9 + group])] = true;
                let r = group / 3 * 3 + offset / 3;
                let c = group % 3 * 3 + offset % 3;
                block[usize::from(solution[r * 9 + c])] = true;
            }
            if !(1..=9).all(|digit| row[digit] && column[digit] && block[digit]) {
                return false;
            }
        }
        true
    }

    #[test]
    fn generated_boards_are_valid_and_have_exact_blank_count() -> Result<()> {
        let generator = SudokuGenerator::new(7, 36)?;
        for id in 0..100 {
            let sample = generator.generate(id);
            assert!(valid(&sample.solution));
            assert_eq!(
                sample.puzzle.iter().filter(|&&digit| digit == 0).count(),
                36
            );
            assert!(
                sample
                    .puzzle
                    .iter()
                    .zip(sample.solution)
                    .all(|(&puzzle, solution)| puzzle == 0 || puzzle == solution)
            );
        }
        Ok(())
    }
}
