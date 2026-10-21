#!/usr/bin/env python
"""Entire parsing of the RRD files from the SWITCH Cricket exports
"""
# ===================
# TODO
# - RRD to xml to csv TODO: all in one function?
# ===================


from pathlib import Path
from datetime import datetime, timedelta
import json
import os
import boto3

import pandas as pd
from rrdxml import dump_csv
import cricket.metadata

# ===================
# PARAMETERS
# testing = True
testing = False
raw_data_dir = Path("/var/lib/cricket")
cur_data_dir = Path("../cricket_dataset/data_current")
out_data_dir = Path("../cricket_dataset")
# ===================


def save_compressed(df,filepath):
    "Save a .tar.gz archive of a DataFrame"

    filename = str(Path(filepath))+ '.tar.gz'
    compression_opts = dict(method='tar',
                            archive_name=filename.name.replace('.tar.gz', ''))
    df.to_csv(
        filename,
        compression=compression_opts,
        index=False)

    return 0


def save_uncompressed(df,filepath):
    "Save the DataFrame as CSV"

    filename = Path(filepath)
    df.to_csv(
        filename,
        index=False)

    return filename


# ===
# Parsing workflow
# 1. Extract RRD metadata
#   - store (if not already present)
#   - update? -> Just overwrite; much simpler
#   - RRD to xml to csv (TODO: all in one function)
# 2. Merge with cumulative data
# 3. Push cumulative data
# 4. Clean-up
# ===


def extract_from_targets(targets):

    # ===
    # 1. From RRD to 'current' CSV

    # logging
    print("\nFrom RRD to 'current' CSV...")

    file_count = 0
    last_update = 0

    for t in targets:

        target_type = t.target_type()
        datasources = t.datasources()
        rras = t.rras()
        rrd_raw = t.rrd_path()

        # create the out data path
        rrd = rrd_raw.relative_to(raw_data_dir)
        cur_data_path = cur_data_dir / rrd.parent
        cur_data_path.mkdir(parents=True, exist_ok=True)

        # keep track of the newest data
        cmd = 'rrdtool last {} > tmp.txt'.format(str(rrd_raw))
        os.system(cmd)
        metadata = int(open('tmp.txt').read())
        last_update = max(last_update, int(metadata))

        rra = rras.index('5minAve')

        # RRD to xml to csv
        # TODO: compact in one function
        cmd = 'rrdtool dump {} {}'.format(str(rrd_raw), 'tmp.xml')
        os.system(cmd)
        csv_file = rrd.stem + '.csv'
        dump_csv('tmp.xml', rra, str(cur_data_path/csv_file), header=datasources)


        # log progress
        file_count +=1
        if file_count%100 == 0:
            print('#file parsed: {} (out of {})'.format(file_count, total_targets))
            if testing: break

    return last_update


def restore_from_bucket(bucket):
    stale_limit = int((datetime.utcnow() - timedelta(1)).strftime("%s"))
    file_count = 0
    for object in bucket.objects.all():
        dst = out_data_dir.joinpath('data_cumulative').joinpath(object.key)
        cur = out_data_dir.joinpath('data_current').joinpath(object.key)
        if dst.exists() and dst.stat().st_mtime >= stale_limit:
            pass
            # print(f"not restoring {object.key}, what we have is new enough")
        elif not cur.exists():
            pass
            # print(f"not restoring {object.key} since there is no newer data")
        else:
            dst_dir = dst.parent
            if not dst_dir.is_dir():
                dst_dir.mkdir(parents=True)
            bucket.download_file(object.key, dst)
            file_count +=1
            if file_count%100 == 0:
                print('#files restored: {}'.format(file_count))
                if testing: break


def cumulate_csv_files():

    csv_files = sorted(cur_data_dir.glob('**/*.csv'))

    print("\nRestoring from object storage...")
    restore_from_bucket(bucket)

    # logging
    file_count = 0
    total_files = len(csv_files)
    print("\nMerging CSVs into 'cumulative'...")
    files = []

    for csv_cur in csv_files:

        # Load latest data file
        df_cur = pd.read_csv(csv_cur)

        # Define the output file path
        csv_cum = Path(str(csv_cur).replace('_current', '_cumulative'))
        # print(csv_cur,csv_cum)

        # If output file exists already, merge latest data in
        if csv_cum.exists():

            # [TESTING] Create fake differences in the data
            if testing: df_cur.timestamp += 300*12*24

            df_cum = pd.read_csv(csv_cum)
            df_cum = pd.concat([df_cum, df_cur], ignore_index=True).drop_duplicates(subset='timestamp',ignore_index=True)
            # save_compressed(df_cum, csv_cum)
            save_uncompressed(df_cum, csv_cum)

        # else, save current as output file
        else:
            csv_cum.parent.mkdir(parents=True, exist_ok=True)
            # save_compressed(df_cur, csv_cum)
            a = save_uncompressed(df_cur, csv_cum)
            print(a)

        files.append(csv_cum)

        # log progress
        file_count +=1
        if file_count%100 == 0:
            print('#file parsed: {} (out of {})'.format(file_count, total_files))
            if testing: break
    return files
# ===

def push_new_data(last_update):
    "Commit and push cumulative data"

    print("\nPush new data")
    last_update_str = datetime.fromtimestamp(last_update).strftime('%Y.%m.%d')
    for csv_cum in csv_cum_files:
        name = str(csv_cum.relative_to(out_data_dir.joinpath("data_cumulative")))
        with open(csv_cum) as f:
            contents = f.read()
        object = bucket.put_object(Body=contents,
                                   Key=name)
    git_command_list = '''git fetch mdpush master
    git add {}
    git commit -m 'New metadata. Last update: {}'
    git push mdpush master
    '''.format('metadata.json', last_update_str)
    print(git_command_list)
    os.system(git_command_list)

def cleanup():
    "Clean up temporary files"

    print("\nCleaning up")
    cleanup_command_list = '''
        rm -f tmp.txt
        rm -f tmp.xml
    '''
    print(cleanup_command_list)
    os.system(cleanup_command_list)

subtrees = [
    'router-interfaces',
    'router-power',
    'routers',
    'cpu-usage',
    'transceiver-monitoring',
    'eci/apollo',
]

oc = cricket.metadata.parse_cricket_configs()

session = boto3.Session(profile_name='NetworkResearchData')
s3 = session.resource('s3')
bucket = s3.Bucket('switchlan-load-timeseries')

rrd_files = []
targets = []
target_types = {}

for t in oc.targets(subtrees):
    name = t.name
    c = t.dirconf
    subdir = c.dir
    if not t.is_mtarget() and t.is_collect():
        tt = t.target_type()
        target_types[tt] = {
            'datasources': t.datasources(),
            'rras': t.rras(),
        }
        router = t.router()
        if router is None:
            print(f"target {subdir}/{name} missing router")
        rras = t.rras()
        rrd_path = t.rrd_path()
        if rrd_path.exists():
            # print("{}/{}".format(subdir, name.lower()))
            rrd_files.append(rrd_path)
            targets.append(t)
        else:
            print(f"missing: {subdir}/{name.lower()}: {rrd_path}")

collected_metadata = {
    'timestamp': datetime.utcnow().isoformat(),
    'targets': [
        { 'dir': str(t.dir()),
          'name': t.name,
          'router': t.router(),
          'target_type': t.target_type(),
         }
        for t in sorted(targets, key=lambda target: str(target.dir()) + '/' + target.name)
    ],
    'targettypes': target_types,
}

with open("metadata.json", "w", encoding="utf-8") as metadata:
    metadata.write(json.dumps(collected_metadata, indent=2))

total_targets = len(targets)

print(f"{total_targets} targets")

last_update = extract_from_targets(targets)
csv_cum_files = cumulate_csv_files()
if not testing:
    push_new_data(last_update)
cleanup()
