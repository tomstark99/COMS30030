if [ -z $2 ]
then
    make
    if [ $1 == "all" ] || [ $1 == "ALL" ]
    then
        ./a.out 0
        ./a.out 1
        ./a.out 2
        ./a.out 3
        ./a.out 4
        ./a.out 5
        ./a.out 6
        ./a.out 7
        ./a.out 8
        ./a.out 9
        ./a.out 10
        ./a.out 11
        ./a.out 12
        ./a.out 13
        ./a.out 14
        ./a.out 15
    else
        ./a.out $1
    fi
else
    make $1
    if [ $2 == "all" ] || [ $2 == "ALL" ]
    then
        ./a.out 0
        ./a.out 1
        ./a.out 2
        ./a.out 3
        ./a.out 4
        ./a.out 5
        ./a.out 6
        ./a.out 7
        ./a.out 8
        ./a.out 9
        ./a.out 10
        ./a.out 11
        ./a.out 12
        ./a.out 13
        ./a.out 14
        ./a.out 15
    else
        ./a.out $2
    fi
fi
