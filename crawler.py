import csv
import requests
from bs4 import BeautifulSoup


# headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36',
    'referer': 'https://google.com/'
}
base_url = "https://www.ratemyprofessors.com/professor/"

# website specific naming
score_div = "CardNumRating__CardNumRatingNumber-sc-17t4b9u-2"
comment_div = "Comments__StyledComments-dzzyvm-0"
course_meta = "CourseMeta__StyledCourseMeta-x344ms-0"

with open('comments.csv',"w", encoding="utf-8", newline="") as csvfile:

    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Comments","Score","Difficulty","Attendance","Would Take Again"])

    for i in range (10235,12000):
        print("Getting professor: " + str(i))
        url = base_url + str(i)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            comment_bodys = soup.find_all('div', class_='Rating__RatingBody-sc-1rhvpxz-0 dGrvXb')
            
            for one_comment in comment_bodys:
                
                child_divs = one_comment.find_all('div')
                comment,score,diff = None,None,None
                attendence,take_again = None, None
                set_score = False
                for div in child_divs:
                    if div['class'][0] == comment_div:
                        comment = div.text
                        # discard null comments
                        if comment == "No Comments":
                            comment = None
                    if div['class'][0] == score_div and not set_score:
                        score = div.text
                        set_score = True
                    if div['class'][0] == score_div and set_score:
                        diff = div.text

                    # course Metadata
                    if div['class'][0] == course_meta:
                        nested_divs = div.find_all('div')
                        for nest_div in nested_divs:
                            parts = nest_div.text.split(":")
                            if parts[0] == "Attendance":
                                attendence = parts[1]
                            if parts[0] == "Would Take Again":
                                take_again = parts[1]
                        
                # comment and quality are required entry
                # others are optional
                # print(comment, score, diff, attendence, take_again)
                if comment is not None and score is not None and comment != "":
                    csvwriter.writerow([comment, score, diff, attendence, take_again])
                
        else:
            print("Professor not found")
