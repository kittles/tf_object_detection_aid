TODO: actually document what this is

# TF Object Detection Model Scaffolding

this is basically to streamline some of the repetitive and brittle
aspects of training object detectors in tensorflow

## project_skeleton.py

when you want to start a new experiment, run this and it will build out the directory
structure.

## workflow.py

when you want to run a training of a model, run this- it generates tf records and scripts
to run a training session. each time it creates a new directory, so that each run can be isolated
and tweaked without affecting any other runs
