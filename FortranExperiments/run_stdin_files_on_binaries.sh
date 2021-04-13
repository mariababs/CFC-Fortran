echo "starting test"
for binary in *.out
do
    echo "Starting"
    echo $binary
    for stdin in stdin*
    do
        rm testout* -f
        echo "stdin"
        echo $stdin
        cat $stdin | ./$binary > $binary$stdin.log
    done
    echo "Done with"
    echo $binary
done
tput bel