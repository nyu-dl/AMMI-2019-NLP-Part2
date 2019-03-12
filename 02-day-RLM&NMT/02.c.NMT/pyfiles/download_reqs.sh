echo 'downloading..'
wget ikulikov.name/nmt_data.tar.gz
wget ikulikov.name/nmt_saved_models.tar.gz
echo 'unpacking..'
tar -xzf nmt_data.tar.gz
tar -xzf nmt_saved_models.tar.gz
echo 'removing tar gz..'
rm nmt_data.tar.gz
rm nmt_saved_models.tar.gz
echo 'done'