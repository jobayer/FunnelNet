def rename_duplicates(files, sep):
    new_list = []
    for i, v in enumerate(files):
        total_count = files.count(v)
        count = files[:i].count(v)
        new_list.append(v + sep + str(count + 1) if total_count > 1 else v)
    return new_list
