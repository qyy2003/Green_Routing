#!/bin/sh
##
## update-topology.sh
##
## Re-generate and push the Switch LAN backbone topology
##
## Date created: 2024-03-18
## Author:       Simon Leinen  <simon.leinen@switch.ch>
##
## This script is intended to be run periodically, for example every
## night.  It can be run as an unattended job, e.g. by cron.
##
## It re-generates the topology files in text and JSON formats,
## including IGP costs and link speeds, as well as information about
## link bundles (in JSON format only).  If anything has changed, the
## output files are committed to Git and pushed to the GitLab server.

SRC=/home/leinen/extract-topology
DST=/home/leinen/network-energy-efficiency-research/switch-network-topology

GIT_ASKPASS=/home/leinen/.ssh/git-mdpush
export GIT_ASKPASS

date="`date +%Y-%m-%d`"

cd "$SRC" || exit 1
test -d .venv || ( echo "No .venv directory found in source directory $SRC"; exit 1 )
. .venv/bin/activate || exit 2

poetry run $SRC/extract_topology/extract_topology.py -c -s \
       -o $DST/switch-network-topology.txt.new || exit 1
poetry run $SRC/extract_topology/extract_topology.py -c -s \
       -f json -o $DST/switch-network-topology.json.new \
       --bundle-output-file $DST/bundles.json.new || exit 1

cd $DST || exit 1

needs_checkin=false
for file in switch-network-topology.txt switch-network-topology.json bundles.json
do
  if test -r $file && diff $file $file.new >$file.diff
  then
    rm $file.new $file.diff
  else
    test -r $file.diff && ls -l $file.diff
    needs_checkin=true
    if test -r $file
    then
      mv $file $file.old || exit 1
    fi
    mv $file.new $file || exit 1
    git add $file || exit 1
    test -r $file.diff && rm $file.diff
  fi
done

if $needs_checkin
then
  echo Checking in...
  git commit -m "Update topology files from router configuration ($date)" || exit 1
  git push mdpush master || exit 1
fi
