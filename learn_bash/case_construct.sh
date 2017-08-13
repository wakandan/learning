if [ -z $1 ]; then
    #if no command line arg
    rental="***Unknown vehicle***"
elif [ -n $1 ]; then
    #make first arg as rental
    rental=$1
fi

echo "Current rental is: $rental"

case $rental in 
    "car")    echo "For $rental Rs.20 per k/m";;
    "van")    echo "For $rental Rs.10 per k/m";;
    "jeep")   echo "For $rental Rs.5 per k/m";;
    "bicyle") echo "For $rental 20 pasa per k/m";;
    *)        echo "Sorry, I can not get a $rental for you";;
esac
