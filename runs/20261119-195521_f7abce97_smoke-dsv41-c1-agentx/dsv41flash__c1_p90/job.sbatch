#!/bin/bash
#SBATCH --job-name=ab-dsv41flash__c1_p90
#SBATCH --partition=amd-spur-v2
#SBATCH --qos=amd-agentx-qos
#SBATCH --account=amd-agentx
#SBATCH --exclude=crsuse2-m2m-v3-177
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=0-02:30:00
#SBATCH --output=/shared_nfs/qiangh/arbor_ab/runs/20261119-195521_f7abce97_smoke-dsv41-c1-agentx/dsv41flash__c1_p90/logs/slurm-%j.out
#SBATCH --error=/shared_nfs/qiangh/arbor_ab/runs/20261119-195521_f7abce97_smoke-dsv41-c1-agentx/dsv41flash__c1_p90/logs/slurm-%j.err
# One PR-test cell: the frozen runner does every phase; nothing else runs on the node.
exec bash /shared_nfs/qiangh/arbor_ab/runs/20261119-195521_f7abce97_smoke-dsv41-c1-agentx/dsv41flash__c1_p90/runner.sh /shared_nfs/qiangh/arbor_ab/runs/20261119-195521_f7abce97_smoke-dsv41-c1-agentx/dsv41flash__c1_p90 2>&1
