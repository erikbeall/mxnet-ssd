
iter=507
files=`\ls *jpg`
prefix="image_"
opts="-resize 500x500^ -gravity center -extent 500x500"
echo ""
echo "starting with iter=$iter"
echo ""
mkdir outputs
for f in $files ; do  
    fnum=`printf %04d $iter`
    convert $opts $f outputs/$prefix$fnum.jpg
    #convert $opts $f ../bbox_labelled/Images/001/$prefix$fnum.jpg
    iter=`expr $iter \+ 1`
done
echo ""
echo "done with iter=$iter"
echo ""

