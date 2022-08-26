
HOME="$1"

python ./pull_data.py

cp $HOME/data/raw /gcsmount-research-data-staging/osmnx-cities