# Accelerating Genomics Analysis with GPUs
## Intro.
![Modern genomics analysis workflow](/assets/img/teaser.png)
While modern genomics analysis promise life-saving innovations in personalized therapeutics and cancer research, their computational pipelines are notoriously slow. The pipelines process millions to billions of DNA sequences read by modern high-throughput sequencing machines, *reads*, to detect patterns. GPUs are natural match to exploit inter- and intra- data parallelism in these pipelines, but the bioinformatics community is slower to adopt GPUs, compared to others like the AI community.

In this project, we build GPU software library to accelerate essential computational blocks in modern genomics pipelines. We target specific gold standard algorithms for practical plug-and-play replacement. In this work, we present **G<sup>3</sup>SA**, the first GPU library covering the established **BWA-MEM** short read aligner end-to-end. Using 4 commodity GPU cards, we demonstrate 70x speedup compared to running BWA-MEM2 on a 12-core desktop CPU.

## Workflow
