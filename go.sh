# $1 $1 or semeval

echo "STEP 1. process raw data"
python process_raw_data.py -h
python -u process_raw_data.py $1
echo -e "\n"

echo "STEP 2. prepare data for classification w/o user clusters"
python process_data.py -h
python classifiers.py -h
python -u process_data.py --use_pv 1 $1
echo -e "\n"

echo "STEP 2a. perform classification w/ unigram"
python -u classifiers.py $1
echo -e "\n"

echo "STEP 2b. perform classification w/ uni/bi gram"
python -u classifiers.py --n_gram 2 $1
echo -e "\n"

echo "STEP 2c. perform classification w/ NBSVM+uni"
python -u classifiers.py --nb 1 $1
echo -e "\n"

echo "STEP 2d. perform classification w/ paragraph vector"
python -u classifiers.py --use_pv 1 $1
echo -e "\n"


echo "STEP 3. prepare data and perform classification w/ user clusters, uni-gram only"
echo "STEP 3a. cluster users with follow graph"
echo "(0) Affinity Propagation" 
python -u process_data.py --graph fnet --clu_algo ap --use_pv 1 $1
python -u classifiers.py --use_cluster 1 $1
python -u classifiers.py --use_cluster 2 $1
python -u classifiers.py --use_pv 1 $1
echo -e "\n"
for clu in 2 5 10 25 50 75 100
do
    echo "(1) Spectral Clustering with cluster number = $clu" 
    python -u process_data.py --graph fnet --clu_algo spectral --n_clu $clu --use_pv 1 $1
    python -u classifiers.py --use_cluster 1 $1
    python -u classifiers.py --use_cluster 2 $1
    python -u classifiers.py --use_pv 1 $1
    echo -e "\n"
    echo "(2) KMeans Clustering with cluster number = $clu" 
    python -u process_data.py --graph fnet --clu_algo kmeans --n_clu $clu --use_pv 1 $1
    python -u classifiers.py --use_cluster 1 $1
    python -u classifiers.py --use_cluster 2 $1
    python -u classifiers.py --use_pv 1 $1
    echo -e "\n"
done

echo "STEP 3b. cluster users with mention graph"
echo "(0) Affinity Propagation" 
python -u process_data.py --graph mnet --clu_algo ap --use_pv 1 $1
python -u classifiers.py --use_cluster 1 $1
python -u classifiers.py --use_cluster 2 $1
python -u classifiers.py --use_pv 1 $1
echo -e "\n"
for clu in 2 5 10 25 50 75 100
do
    echo "(1) Spectral Clustering with cluster number = $clu" 
    python -u process_data.py --graph mnet --clu_algo spectral --n_clu $clu --use_pv 1 $1
    python -u classifiers.py --use_cluster 1 $1
    python -u classifiers.py --use_cluster 2 $1
    python -u classifiers.py --use_pv 1 $1
    echo -e "\n"
    echo "(2) KMeans Clustering with cluster number = $clu" 
    python -u process_data.py --graph mnet --clu_algo kmeans --n_clu $clu --use_pv 1 $1
    python -u classifiers.py --use_cluster 1 $1
    python -u classifiers.py --use_cluster 2 $1
    python -u classifiers.py --use_pv 1 $1
    echo -e "\n"
done

echo "STEP 3c. cluster users with full follow graph"
for clu in 2 5 10 25 50 75 100
do
    echo "(1) KMeans Clustering with cluster number = $clu" 
    python -u process_data.py --graph fnetf --clu_algo kmeans --n_clu $clu --use_pv 1 $1
    python -u classifiers.py --use_cluster 1 $1
    python -u classifiers.py --use_cluster 2 $1
    python -u classifiers.py --use_pv 1 $1
    echo -e "\n"
done

echo "STEP 3d. cluster users with full mention graph"
for clu in 2 5 10 25 50 75 100
do
    echo "(1) KMeans Clustering with cluster number = $clu" 
    python -u process_data.py --graph mnetf --clu_algo kmeans --n_clu $clu --use_pv 1 $1
    python -u classifiers.py --use_cluster 1 $1
    python -u classifiers.py --use_cluster 2 $1
    python -u classifiers.py --use_pv 1 $1
    echo -e "\n"
done


