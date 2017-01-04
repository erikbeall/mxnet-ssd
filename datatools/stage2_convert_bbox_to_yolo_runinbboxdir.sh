#!/bin/bash

calc(){ 
    awk "BEGIN { print "$*" }"  
}

bdir=`pwd`
cnum=5
bnum=`expr $cnum \- 1`
subdir=`printf %03d $cnum`
cd Labels/$subdir
files=`\ls *txt`

output_training_file="data_list_train.txt"
cd $bdir

for f in $files ; do
    infile=$f
    infile=Labels/$subdir/$infile
    # location of new files in annotations subdirectory
    outfile="annotations/$f"
    #outfile=labels/annotations/$outfile
    # get the corresponding image filename
    imagefile=`echo $f | sed "s/.txt/.jpg/"`
    imagefile="$bdir/Images/$subdir/$imagefile"
    # test if image exists
    if [ ! -f $imagefile ] ; then
        echo "skipping file $imagefile"
        # remove text file (dangerous but if the image doesnt exist ...)
        rm $infile
        continue
    fi
    alreadypresent=0
    # test if there is already an annotation file present - if so, we don't want to add the filename twice
    if [ -f $outfile ] ; then
        alreadypresent=1
    fi
    a=(`identify $imagefile`)
    dims=(`echo "${a[2]}" | sed "s/x/ /"`)
    # get the x,ys from the file
    addthisfile=0
    while read line; do
        la="${line}"; 
        laa=($la);
        if [ "${#laa[*]}" -eq 4 ] ; then
            x1r=`calc ${laa[0]}+${laa[2]}`
            x1r=`calc $x1r/${dims[0]}`
            x1r=`calc $x1r/2`
            
            y1r=`calc ${laa[1]}+${laa[3]}`
            y1r=`calc $y1r/${dims[1]}`
            y1r=`calc $y1r/2`

            wr=`calc ${laa[2]}-${laa[0]}`
            wr=`calc $wr/${dims[0]}`
            hr=`calc ${laa[3]}-${laa[1]}`
            hr=`calc $hr/${dims[1]}`
            echo $bnum $x1r $y1r $wr $hr >> $outfile
            addthisfile=1
        fi
    done < $infile
    if [ $alreadypresent -eq 1 ]; then
        addthisfile=0
        alreadypresent=0
    fi
    if [ $addthisfile -eq 1 ] ; then
        echo $imagefile >> $output_training_file
        addthisfile=0
    fi
    echo "done with $imagefile, dims = ${dims[0]}x${dims[1]}"
done

