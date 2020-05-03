for i in {1..10}
do 
    python run_attack_classification.py mr_${i}
    mkdir adv_results/mr_${i}
    mv adv_results/adversaries.txt adv_results/mr_${i}/
    mv adv_results/results_log adv_results/mr_${i}/
done

for i in {1..25}
do 
    python run_attack_classification.py imdb_${i}
    mkdir adv_results/imdb_${i}
    mv adv_results/adversaries.txt adv_results/imdb_${i}/
    mv adv_results/results_log adv_results/imdb_${i}/
done