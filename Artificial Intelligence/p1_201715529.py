from datetime import datetime
import random

# 경기 결과 입력 받는 곳
place = input("경기가 열린 곳은? ")
time = input("경기가 열린 시간은? ")
opponent = input("상대 팀은? ")
goals = input("손흥민은 몇 골을 넣었나요? ")
aids = input("도움은 몇 개인가요? ")
score_me = input("손흥민 팀이 넣은 골 수는? ")
score_you = input("상대 팀이 넣은 골 수는? ")

# 기사 작성하는 곳
news = "[프리미어 리그 속보("+str(datetime.now())+")]\n"
news = news+"손흥민 선수는 "+place+"에서 "+time+"에 열린 경기에 출전하였습니다. "
news = news+"상대 팀은 "+opponent+"입니다. "

if score_me > score_you:
    winnig_wording = ["골을 넣은 상대 팀을 이겼습니다.", "상대를 대상으로 통쾌한 승리를 거머쥐었습니다.", "상대 팀을 꺾었습니다.",
                      "승점 3점 획득에 성공합니다."]
    news = news+"손흥민 선수의 팀이 "+score_me+"골을 넣어 "+score_you+random.choice(winnig_wording)
elif score_me < score_you:
    lose_wording = ["골을 넣은 상대 팀에게 졌습니다.", "골로 "+place+"에서 경기 종료됩니다", "완패를 당하며 경기 종료"]
    news = news+"손흥민 선수의 팀이 "+score_me+"골을 넣었지만 "+score_you+random.choice(lose_wording)
else:
    draw_wording = ["로 비겼습니다.", "(으)로 경기 마무리 됩니다.", "양팀 사이 좋게 승점 1점씩 가져갑니다."]
    news = news+"두 팀은 "+score_me+"-"+score_you+random.choice(draw_wording)

if int(goals) > 0 and int(aids) > 0:
    news = news+"손흥민 선수는 "+goals+"골에 도움 "+aids+"개로 승리를 크게 이끌었습니다. "
elif int(goals) > 0 and int(aids) == 0:
    news = news+"손흥민 선수는 "+goals+"골로 승리를 이끌었습니다. "
elif int(goals) == 0 and int(aids) > 0:
    news = news+"손흥민 선수는 골은 없지만 도움 "+aids+"개로 승리하는 데 공헌하였습니다. "
else:
    news = news+"아쉽게도 이번 경기에서 손홍민의 발끝은 침묵을 지켰습니다. "

print(news)