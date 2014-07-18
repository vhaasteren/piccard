#! /bin/sh
# arguments
# 1) input filename
# 2) number of lines to delete (burnin)
# 3) chain thinning number
# 4) output filename

sed '1,'"$2"'d' $1 | awk 'NR%'"$3"'==0' > $4
