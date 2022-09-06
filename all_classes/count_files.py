import os
from classes import paths_other, paths_passport, paths_pts, all_classes


def count_samples_all(where: str) -> (dict, dict):
    """ max and min scans in samples folder"""

    def count_saplmes_dir(readp, directory, cdic) -> dict:
        if os.path.exists(readp):
            for filename in os.listdir(readp):
                if filename.endswith(".png") or filename.endswith(".jpg"):  # photo
                    if directory in cdic:
                        cdic[directory] += 1
                    else:
                        cdic[directory] = 1
        return cdic

    cdic = dict()
    full_passport_path = ['passport/' + vv for vv in paths_passport]
    full_pts_path = ['pts/' + vv for vv in paths_pts]
    full_path = dict()

    dirs = full_passport_path + full_pts_path + paths_other

    for directory in dirs:
        # detect global class
        cl_now = None
        for x in all_classes:
            if x in directory:
                cl_now = x

        if cl_now not in full_path:
            full_path[cl_now] = list()  # sorted
        if where == './samples/':
            for i in range(4):  # 0,1,2,3
                readp = where + directory + '/' + str(i)
                cdic = count_saplmes_dir(readp, directory, cdic)
                full_path[directory].append(readp)
        elif where == './prep/':  # to fineprep
            readp = where + directory
            cdic = count_saplmes_dir(readp, directory, cdic)
            full_path[cl_now].append(readp)

    passports = 0
    ptss = 0
    delk = []
    for k, v in cdic.items():
        if 'passport/' in k:
            passports += v
            delk.append(k)
        if 'pts/' in k:
            ptss += v
            delk.append(k)

    for x in delk:
        del cdic[x]

    cdic['passport'] = passports
    cdic['pts'] = ptss

    return cdic, full_path