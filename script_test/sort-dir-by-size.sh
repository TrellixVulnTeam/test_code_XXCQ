# sort-dir-by-size.sh
#!/bin/bash 
 
if [ $# != 2 ]; then 
	echo "Incorrect number of arguments !" >&2 
	echo "USAGE: sortdirbysize [DIRECTORY] <first n directories>" 
fi 
du -m -d 1 $1 | sort -rn | head -$2