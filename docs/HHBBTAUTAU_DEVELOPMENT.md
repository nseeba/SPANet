# SPANet Development for VBF HH -> bbtautau

This document records the current SPANet development state for the VBF HH -> bbtautau analysis. It is intended as a handover and reference for future training and evaluation.

## Scientific Goal

The model must assign two different jet pairs in each event:

- `higgs_bb`: the two b-jets from the hadronic Higgs decay;
- `vbf`: the two VBF tagging jets.

The main performance requirement is not merely to reproduce the existing two-task SPANet baseline. The important target is the performance of a dedicated HH-only SPANet model, which reached approximately `96.7%` HH exact-pair efficiency. The combined model should retain this HH performance while also predicting the VBF pair.

The analysis baseline is the existing HH-b-tag/VBF-tagger based reconstruction. SPANet should be compared with that baseline using the same event sample, truth definition, validity requirements, and pair-matching convention.

## Model Structure

SPANet is a shared jet-reconstruction network with branch-specific decoders. In this setup it has:

1. Input embeddings for the configured jet and event features.
2. A shared `JetEncoder` transformer-style representation.
3. One branch decoder for `higgs_bb`.
4. One branch decoder for `vbf`.
5. Detection and assignment outputs for each branch.

The model is therefore a multi-task architecture. Both tasks use the shared embedding and encoder, so updates from the VBF task can change the representation used by the HH task. This shared representation is the reason why a model can perform well on each task separately but lose HH performance when both tasks are trained together.

The current five-feature configuration is defined in:

- `event_files/vbf_hh_bbtautau_5features.yaml`
- `options_files/vbf_hh_bbtautau/train.json`
- `options_files/vbf_hh_bbtautau/pcgrad.json`

The b-tag feature configuration is separate:

- `event_files/vbf_hh_bbtautau_btag_features.yaml`

## Data and Training Configuration

The main signal training uses:

```text
/local/norman/UHHAnalysis/SPANet/hdf5/22_23_v14_vbf_signal_train.h5
/local/norman/UHHAnalysis/SPANet/hdf5/22_23_v14_vbf_signal_val.h5
```

The common test sample is:

```text
/local/norman/UHHAnalysis/SPANet/hdf5/22_23_v14_vbf_signal_test_23only.h5
```

The standard comparison settings are batch size `512`, `50` epochs, and the five-feature event file. On Manivald, training is submitted through the Slurm launchers in `scripts/`. The PCGrad launcher currently requests `96:00:00`; this was increased because previous jobs were stopped at the 48-hour limit.

Always verify the run name, checkpoint path, event file, batch size, precision, and Slurm time limit in the output log before starting a long training.

## Decoder Validity Fix

The original decoder could select the same jet twice, producing assignments such as `(j, j)`. This affected both HH and VBF predictions and made the original prediction files unsuitable for final physics comparisons.

The current decoder masks diagonal pair assignments for two-parton targets. It also masks the jets selected by the HH branch before selecting the VBF pair. The relevant logic is in:

```text
spanet/network/prediction_selection.py
```

Every new prediction file must be checked for:

- duplicate HH pair indices;
- duplicate VBF pair indices;
- overlap between the selected HH and VBF jets;
- out-of-range or incomplete predictions when an event has too few usable jets.

The incomplete-event count is not automatically a bug. It can occur when the event has fewer than the required number of valid jets. Duplicate pairs or HH/VBF overlap on valid events are bugs.

The evaluation script uses the current decoder code at evaluation time. Prediction files should therefore be regenerated after decoder changes rather than reusing files made with the old selector.

## Loss Balancing and PCGrad

The standard model has configurable loss scales and automatic loss balancing. The relevant options include:

```text
balance_losses
higgs_bb_assignment_loss_scale
vbf_assignment_loss_scale
higgs_bb_detection_loss_scale
vbf_detection_loss_scale
```

The weighting scans varied task or branch loss scales while keeping the data, architecture, batch size, and training duration fixed. They showed that simple scaling did not restore the HH-only performance.

PCGrad is enabled with:

```json
"pcgrad": true
```

When PCGrad is enabled, automatic Lightning optimization is disabled and the training step computes separate gradients for the HH and VBF task losses. The current implementation:

1. Computes the HH assignment plus detection loss.
2. Computes the VBF assignment plus detection loss.
3. Flattens the task gradients over the model parameters.
4. Measures their dot product and cosine similarity.
5. If the gradients conflict, projects the VBF gradient to remove its component opposing the HH gradient.
6. Leaves the HH gradient unchanged.
7. Merges the gradients and performs one optimizer step.

This is a protected, asymmetric PCGrad variant. It is not the fully symmetric textbook PCGrad algorithm: HH is treated as the protected task because the HH-only result is the primary target.

The implementation and diagnostics are in:

```text
spanet/network/jet_reconstruction/jet_reconstruction_training.py
```

The PCGrad configuration is:

```text
options_files/vbf_hh_bbtautau/pcgrad.json
```

For PCGrad runs, `balance_losses` is disabled. The task weights are currently both `1.0`, so the experiment tests gradient conflict handling rather than another manually chosen loss ratio.

## Completed Models and Results

The results below come from the unique-decoder evaluation notebook:

```text
notebooks/vbf_hh_bbtautau_model_evaluation_unique_decoder.ipynb
```

The main exact-pair efficiencies are:

| Model | HH | VBF | Both |
|---|---:|---:|---:|
| Balanced two-task baseline | 93.944% | 88.493% | 94.405% |
| No loss balancing | 93.704% | 88.271% | 94.706% |
| HH assignment scale 1.25 | 91.640% | 87.928% | 93.296% |
| HH assignment scale 1.50 | 93.715% | 87.175% | 94.351% |
| Earlier PCGrad | 91.490% | 85.165% | 92.834% |
| PCGrad v2 | 91.297% | 85.861% | 93.039% |
| Global protected PCGrad | 93.946% | 87.480% | 94.398% |
| HH-only SPANet | approximately 96.7% | not applicable | not applicable |

The global protected-PCGrad model successfully restores HH performance to the ordinary balanced two-task baseline, but it does not reach the `~96.7%` HH-only target. Its VBF performance is also about one percentage point below the balanced two-task model.

The current prediction validity checks for the global-PCGrad file show:

- zero duplicate HH pairs;
- zero duplicate VBF pairs;
- zero HH/VBF selected-jet overlap;
- zero invalid truth pairs.

The prediction files are therefore structurally valid after the unique-decoder fix. The remaining problem is model performance, not pair-format validity.

## TensorBoard Diagnostics

The loss and gradient comparison notebook is:

```text
notebooks/SPANet_tensorboard_loss_scan.ipynb
```

It compares the HH-only reference, balanced baseline, loss-scan models, and PCGrad models. Important PCGrad tags are:

```text
pcgrad/task_cosine
pcgrad/conflict
pcgrad/projection_applied
pcgrad/hh_gradient_norm
pcgrad/vbf_gradient_norm
pcgrad/weight_higgs_bb
pcgrad/weight_vbf
```

Interpretation:

- Frequent negative cosine values and projections indicate genuine gradient conflict.
- A large VBF gradient norm relative to the HH norm indicates that VBF can dominate shared-parameter updates.
- If conflict is frequent but HH remains below `96%`, projecting the VBF gradient is not sufficient to preserve the HH representation.
- If conflict is rare, the performance gap is likely caused by shared representation capacity, task-specific ambiguity, or the data/truth definition rather than direct gradient opposition.

The TensorBoard diagnostics should be interpreted together with the held-out exact-pair efficiencies. Lower training loss or higher validation accuracy alone is not enough because the generic validation accuracy is not the primary physics metric.

## Current Conclusions

1. The old duplicate-jet issue has been fixed in the current decoder and regenerated prediction files.
2. Simple loss scaling did not recover HH-only performance.
3. Earlier PCGrad implementations degraded both tasks.
4. Global protected PCGrad prevents the HH degradation seen in earlier PCGrad runs and returns HH to approximately `93.9%`.
5. It still does not preserve the approximately `96.7%` HH-only result.
6. The unresolved issue is the interaction between the shared representation and the VBF task, not the validity of the selected jet pairs.

## Recommended Next Steps

### 1. Finish the diagnostic comparison

Run all cells in `SPANet_tensorboard_loss_scan.ipynb` with the HH-only and global-PCGrad runs included. Record the conflict fraction, projection fraction, task cosine, and gradient-norm ratio.

### 2. Add a shared-versus-task-specific ablation

The next architectural test should keep the input features, data, decoder validity rules, batch size, and number of epochs fixed while changing only the gradient routing or network sharing. Useful controlled variants are:

- HH-preserving shared encoder with a task-specific VBF adapter;
- a partially shared encoder with separate late layers for HH and VBF;
- stop-gradient from the VBF branch into selected shared layers;
- a symmetric PCGrad variant as a comparison, while retaining protected PCGrad as the primary hypothesis.

The HH-only model should remain the reference, not a required pretraining checkpoint. A successful architecture should reach close to the HH-only result from a single combined training run.

### 3. Keep evaluation fixed

For every new model, use the same test file and the same evaluation notebook. Report at minimum:

- HH exact-pair efficiency;
- VBF exact-pair efficiency;
- simultaneous HH and VBF efficiency;
- duplicate-pair and cross-branch-overlap rates;
- performance as a function of jet multiplicity;
- score-threshold curves.

Only after the single-run combined model reaches the HH target should additional training samples such as ggF HH or semileptonic ttbar, or additional feature sets, be introduced. Those are separate ablations and should not be mixed into the architectural comparison.

## Useful Files

```text
spanet/network/jet_reconstruction/jet_reconstruction_training.py
spanet/network/prediction_selection.py
spanet/network/jet_reconstruction/jet_reconstruction_network.py
spanet/options.py
options_files/vbf_hh_bbtautau/pcgrad.json
event_files/vbf_hh_bbtautau_5features.yaml
scripts/train_vbf_hh_bbtautau_pcgrad_gpu.sh
scripts/evaluate_vbf_hh_bbtautau_gpu.sh
notebooks/vbf_hh_bbtautau_model_evaluation_unique_decoder.ipynb
notebooks/SPANet_tensorboard_loss_scan.ipynb
```
