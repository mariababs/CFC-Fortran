echo "starting test"
rm testout* -f
cat small_timestep_stdin.txt | ./2ndOrderOmegaCalc.out > second_order_small_timestep.log
echo "section 1 done"
rm testout* -f
#cat large_timestep_stdin.txt | ./4thOrderOmegaCalc.out > fourth_order_large_timestep.log
echo "section 2 done"
