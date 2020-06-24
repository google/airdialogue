num_turns=6
max_turn_n=$(expr $num_turns - 1)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

datajson=$1
kbjson=$2
model=$3

python $DIR/initialize.py $datajson

masterjson='start.data.json'
masterjson_alt='start.data.json.1'
cp $masterjson $masterjson_alt
t_masterjson='next.data.json'
agentjson='agent.data.json'
clientjson='client.data.json'

agentout='agentout.txt'
clientout='clientout.txt'

echo 'Running agent first flow...'

for i in $(eval echo {0..$max_turn_n});
do
	# Make agent first flow
	# a_i_a
	python $DIR/clean_for_agent.py $masterjson
	bash $model/scripts/codalab_selfplay_step.sh $agentout $agentjson $kbjson
	python $DIR/clean_after_agent.py $masterjson $agentout $t_masterjson
	rm $agentout $masterjson
	mv $t_masterjson $masterjson
	# a_i_c
	python $DIR/clean_for_client.py $masterjson
	bash $model/scripts/codalab_selfplay_step.sh $clientout $clientjson
	python $DIR/clean_after_client.py $masterjson $clientout $t_masterjson
	rm $clientout $masterjson
	mv $t_masterjson $masterjson
done;

echo 'Running client first flow...'

mv $masterjson agent_first.json
mv $masterjson_alt $masterjson

for i in $(eval echo {0..$max_turn_n});
do
	# Make client first flow
	# c_i_c
	python $DIR/clean_for_client.py $masterjson
	bash $model/scripts/codalab_selfplay_step.sh $clientout $clientjson
	python $DIR/clean_after_client.py $masterjson $clientout $t_masterjson
	rm $clientout $masterjson
	mv $t_masterjson $masterjson
	# c_i_a
	python $DIR/clean_for_agent.py $masterjson
	bash $model/scripts/codalab_selfplay_step.sh $agentout $agentjson $kbjson
	python $DIR/clean_after_agent.py $masterjson $agentout $t_masterjson
	rm $agentout $masterjson
	mv $t_masterjson $masterjson
done;

rm $agentjson $clientjson
mv $masterjson client_first.json

