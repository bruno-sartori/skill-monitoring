#!/bin/sh

for i in "$@"
do
case $i in
	-h|--help)
	echo "Usage: sh run.sh --mode=development|production --execute=tracker|facenet|api"
	echo "api optional parameters: --api-host=http://localhost:3000 --api-port=3000"
	exit 0;
	;;

	-m=*|--mode=*)
    mode="${i#*=}"
    ;;

    -e=*|--execute=*)
    execute="${i#*=}"
    ;;

	-p=*|--api-port=*)
    port="${i#*=}"
    ;;

	-h=*|--api-host=*)
    host="${i#*=}"
    ;;

	*)
            # unknown option
    ;;
esac
done

PYTHON_ENV=$mode CORE_HOST=http://localhost:3000 python init.py --module $execute --host=$host --port=$port
