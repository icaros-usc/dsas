for TRIAL in {1..5}
do
  python search/search.py -t $TRIAL -c search/config/experiment/experiment.tml & 
done
