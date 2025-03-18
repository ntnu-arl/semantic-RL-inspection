#!/usr/bin/env bash
export SEMANTIC_RL_INSPECTION_DIR=$HOME/workspaces/semantic-RL-inspection/
python3 -m rl_training.enjoy_semanticRLinspection_semantic_and_obstacles_example --train_dir=$SEMANTIC_RL_INSPECTION_DIR/examples/pre-trained_network --experiment=default_network --env=inspection_task  --load_checkpoint_kind=best