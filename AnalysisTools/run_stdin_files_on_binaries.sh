echo "starting test"
for binary in *.out
do
    for stdin in stdin*
    do
        rm testout* -f
        cat $stdin | ./$binary > $binary$stdin.log
    done
done
tput bel