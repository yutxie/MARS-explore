# Explorative MARS (Markov Molecular Sampling for Multi-objective Drug Discovery)

Thanks for your interest! This is the code repository for the explorative version of the MARS algorithm, as introduced in the ICLR 2023 paper [How Much Space Has Been Explored? Measuring the Chemical Space Covered by Databases and Machine-Generated Molecules](https://openreview.net/forum?id=Yo06F8kfMa1), where we explicitly encourage the model to explore the unknown chemical space by incorporating coverage measures into the optimization objective. 

For the original definition and implementation of MARS, please visit the [paper](https://openreview.net/pdf?id=kHSu4ebxFXY) and the [repo](https://github.com/bytedance/markov-molecular-sampling). For the implementation of chemical space measures, please visit this [repo](https://github.com/yutxie/chem-measure). 

## Usage

```
python -m MARS.main --train --run_dir $RUN_DIR --objectives $OBJECTIVES --nov_term $NOV_TERM --nov_coef $NOV_COEF
```

Example:
```
python -m MARS.main --train --run_dir ./runs/example --objectives jnk3,qed,sa --nov_term NCircles --nov_coef 0.1
```
