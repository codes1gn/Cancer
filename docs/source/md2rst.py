import os
import json
import requests
import shutil
import glob




def md2rst(md_files: list, rst_file: str) -> str:

    rst_string = str()
    
    rst_string += ("=" * 30 + "\nNumpy\n" + "=" * 30 + "\n\n")
    print(len(md_files))
    for md_file in md_files:
        md_file_name = md_file.split("/")[-1].split(".md")[0]
        rst_string += (md_file_name + "\n" + "=" * 30 + "\n")
        print(md_file_name)

        response = requests.post(url='http://c.docverter.com/convert',
                                 data={
                                     'to': 'rst',
                                     'from': 'markdown'
                                 },
                                 files={'input_files[]': open(md_file, 'rb')})
        if response.ok:
            rst_string += ("\n" + response.content.decode("utf-8") + "\n")

    with open(rst_file, "w") as f:
        f.write(rst_string)

    return rst_string


if __name__ == "__main__":
    md_files = glob.glob("./Numpy/*.md")
    rst_file = "./Numpy/index.rst"
    print(md_files[0])
    md2rst(md_files=md_files, rst_file=rst_file)