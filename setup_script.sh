# cd to the root directory of the project
cd /project_antwerp/MD_DL_DPP8-9 || exit
git config --global --add safe.directory /project_antwerp/MD_DL_DPP8-9
token=$(cat '/project_antwerp/token.txt')
echo "machine github.com login oauth2 password ${token}" >> ~/.netrc
export HF_HOME=/project_antwerp/temp/hf_cache
git pull