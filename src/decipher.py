block_index = 0
blocks = []
dict_guitar_strings = {"e": 0, "B": 1, "G": 2, "D": 3, "A": 4, "E": 5}
last = []

def main(filename):
    global blocks
    with open(filename, "r") as file:
        blocks_txt = file.read()

    blocks = blocks_txt.split("\n\n")
    blocks.pop()
    deci()
    return last

def sort(arr, index):
    for i in range(0, len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i][index] > arr[j][index]:
                temp = arr[i]
                arr[i] = arr[j]
                arr[j] = temp

def calc_point(note):
    return dict_guitar_strings[note[0]] * 19 + note[1]

def deci():
    extracted = []
    temp_arr = []
    fret_groups = []
    end_product = []

    global block_index, last
    text = blocks[block_index]
    lines = text.split("\n")
    non_empty_lines = [line for line in lines if line.strip() != ""]

    for line in non_empty_lines:
        frets = line.split("|")
        frets.pop()
        for i in range(1, len(frets)):
            for j in range(0, len(frets[i])):
                if frets[i][j].isdigit():
                    extracted.append([frets[0], int(frets[i][j]), i, j])

    sort(extracted, 2)

    for i in range(0, len(extracted)):
        if i != len(extracted) - 1:
            if extracted[i][2] != extracted[i+1][2]:
                temp_arr.append(extracted[i])
                fret_groups.append(temp_arr.copy())
                temp_arr.clear()
                continue
            temp_arr.append(extracted[i])
        else:
            temp_arr.append(extracted[i])
            fret_groups.append(temp_arr.copy())
            temp_arr.clear()
            break

    for fg in fret_groups:
        sort(fg, 3)
        for f in fg:
            f.append(calc_point(f))
            del f[:2]


    for fg in fret_groups:
        for i in range(0, len(fg)):
            if i != len(fg) - 1:
                if fg[i][1] != fg[i+1][1]:
                    temp_arr.append(fg[i][2])
                    end_product.append(temp_arr.copy())
                    temp_arr.clear()
                    continue
                temp_arr.append(fg[i][2])
            else:
                temp_arr.append(fg[i][2])
                end_product.append(temp_arr.copy())
                temp_arr.clear()
                break

    last.extend(end_product)
    block_index += 1
    if block_index != len(blocks):
        deci()

