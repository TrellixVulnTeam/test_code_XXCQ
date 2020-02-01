# 本文档用于gnuplot绘制曲线
# Set the output to a png file
set terminal png size 1000,1000

# The file we'll write to
set output 'curve.png'

# The graphic title
set title 'Fitted Curve'
set xlabel "x"
set ylabel "y"

#plot all curves
set style line 5 lt rgb "red" lw 1 pt 1
plot "./build/curve_data.dat" using 1:2 title "noisy points", \
     exp(1.16608 * x**2 + 1.7883 * x + 1.05531) title "fitting curve" ,\
     exp(1.0 * x**2 + 2.0 * x + 1.0) title "original curve" ls 5
