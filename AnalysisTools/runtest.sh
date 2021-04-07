echo "starting test"
rm testout*
cat stdin | ./2ndOrderOmegaCalc.out > t1.log
rm testout*
cat stdin | ./4thOrderOmegaCalc.out > t2.log

cat t1.log| grep -e "41  T" > t1_short.log
cat t2.log| grep -e "41  T" > t2_short.log
diff t1_short.log t2_short.log
