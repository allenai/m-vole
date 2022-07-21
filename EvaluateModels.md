#TODO do this file
# Evaluating the Pre-trained Models
## Download the Weights
You can download the weights from [here](https://drive.google.com/file/d/15JgEh8xoUy26azUOIPDVIrUfi-X3aufc/view?usp=sharing).
After downloading the weights, extract them to `pretrained_models/saved_checkpoints`.
## Evaluating the Trained Models
### Table 2
#### Row 1 - No Mask Baseline
```
allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/no_mask_baseline --eval -c pretrained_models/saved_checkpoints/no_mask_baseline.pt
```

#### Row 2 - MDM w/ Prediction
```
allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/mdm_with_predicted_mask --eval -c pretrained_models/saved_checkpoints/mdm_with_prediction.pt
```

#### Row 3 - Loc-MDM w/ Prediction
```
allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/loc_mdm_with_predicted_mask --eval -c pretrained_models/saved_checkpoints/loc_mdm_with_prediction.pt
```

#### Row 4 - m-VOLE w/ Prediction
```
allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/mvole_with_predicted_mask --eval -c pretrained_models/saved_checkpoints/mvole_with_prediction.pt
```

#### Row 5 - MDM w/ GT Mask
```
allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/mdm_gt_mask --eval -c pretrained_models/saved_checkpoints/mdm_w_gt_mask.pt
```
#### Row 6 - Loc-MDM w/ GT Mask
```
allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/loc_mdm_w_gt_mask --eval -c pretrained_models/saved_checkpoints/loc_mdm_w_gt_mask.pt
```
#### Row 7 - ArmPointNav
```
allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/armpointnav_baseline --eval -c pretrained_models/saved_checkpoints/armpointnav.pt
```
#### Row 8 - m-VOLE w/ GT Mask
```
allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/mvole_w_gt_mask --eval -c pretrained_models/saved_checkpoints/mvole_w_gt_mask.pt
```

### Table 3 - m-VOLE w/ Depth Noise

```
allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/mvole_with_depth_noise --eval -c pretrained_models/saved_checkpoints/mvole_with_prediction.pt
```

### Fig 5 - m-VOLE w/ Noise in Agent's movements

```
allenact manipulathor_baselines/bring_object_baselines/experiments/ithor/mvole_with_agent_motion_noise --eval -c pretrained_models/saved_checkpoints/mvole_with_prediction.pt
```