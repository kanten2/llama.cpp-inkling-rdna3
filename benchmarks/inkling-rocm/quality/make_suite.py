"""Generate the fixed Japanese questions and independently computed answer key."""
import itertools
import argparse
import json
import sqlite3
from fractions import Fraction
from pathlib import Path

rows = []
def add(id, title, question, expected):
    shape = "整数を1個" if type(expected) is int else "指定順の整数のJSON配列を1個"
    instruction = f"最終回答には{shape}だけを書いてください。"
    rows.append(dict(id=id, title=title, prompt=question + "\n" + instruction, expected=expected))

add("M1","値引きと端数","1冊487円の本を7冊買う。本の小計を15%引きにして1円未満を切り捨て、その金額に8%の税を乗せて再び1円未満を切り捨てる。さらに非課税の送料210円を加える。最終支払額は何円か。", (487*7*85//100)*108//100 + 210)
add("M2","溶液の置換","濃度12重量%の食塩水600gをよく混ぜる。150gを取り除いて水150gを足し、完全に混ぜる。次に200gを取り除いて水200gを足す。容器に残る食塩は何gか。", int(72*Fraction(450,600)*Fraction(400,600)))
balls = "RRRRRBBBBGGG"
draws = list(itertools.combinations(balls,2))
prob = Fraction(draws.count(("B","B")), sum("B" in draw for draw in draws))
add("M3","条件付き確率","赤5個、青4個、緑3個の玉から、無作為に2個を戻さず取り出した。『少なくとも1個は青』だと分かっている。2個とも青である条件付き確率を既約分数の [分子,分母] で答えよ。", [prob.numerator,prob.denominator])
add("M4","制約付き数え上げ","長さ9の0・1列で、1をちょうど4個含み、連続する111を一度も含まないものは何通りか。先頭の0も許す。", sum(s.count("1")==4 and "111" not in s for s in map("".join,itertools.product("01",repeat=9))))

orders=[p for p in itertools.permutations(range(1,6)) if p.index(4)==p.index(2)+1 and p.index(1)<p.index(5) and p.index(5)==p.index(3)+2 and p.index(2)<p.index(1)]
assert len(orders)==1
add("L1","順序制約","作業1から5を各1回、5枠に並べる。4は2の直後。1は5より前。3は5のちょうど2枠前。2は1より前。唯一の並びを先頭から番号の配列で答えよ。",list(orders[0]))
add("L2","真偽条件","整数xは1以上12以下。『xは偶数』『xは3の倍数』『xは7より大きい』『xは素数』の4文のうちちょうど2文が真になるxをすべて昇順に答えよ。", [x for x in range(1,13) if sum((x%2==0,x%3==0,x>7,x in (2,3,5,7,11)))==2])
add("L3","規則の反例","各カードの片面は整数、反対面は赤か緑。『数字が3の倍数なら、色は緑』という規則の違反がないか調べる。今見える面は番号1が6、番号2が8、番号3が緑、番号4が赤、番号5が12。必ず裏返して調べる必要のあるカード番号すべてを昇順に答えよ。",[1,4,5])
edges={(1,2):8,(1,3):12,(2,3):2,(2,4):10,(3,4):4,(3,5):16,(4,5):6,(2,5):18}
cost=[]
for n in range(4):
    for middle in itertools.combinations([2,3,4],n):
        path=(1,*middle,5)
        pairs=list(zip(path,path[1:]))
        if all(edge in edges for edge in pairs):
            w=[edges[edge] for edge in pairs]
            cost.append(sum(w)-max(w)//2)
add("L4","割引付き最短路","有向辺の時間は 1→2:8、1→3:12、2→3:2、2→4:10、3→4:4、3→5:16、4→5:6、2→5:18。クーポン1枚で通る辺1本の時間を半分にできる。1から5までの最短時間はいくつか。",min(cost))

a=[[1]]*3
b=a[:]
b[1].append(2)
a[0]=[9]
b[2]=b[2]+[3]
add("C1","Pythonの参照","Python 3で次を実行した直後の [sum(a[0]),sum(a[1]),sum(a[2]),sum(b[2])] は何か。\na=[[1]]*3\nb=a[:]\nb[1].append(2)\na[0]=[9]\nb[2]=b[2]+[3]",list(map(sum,(a[0],a[1],a[2],b[2]))))
a,b,s=2,5,0
for i in range(1,8):
    a,b=b,(a+2*b+i)%11
    if b%2: s+=a
add("C2","同時代入とループ","Python 3で次を実行する。最後の [a,b,s] を答えよ。\na,b,s=2,5,0\nfor i in range(1,8):\n    a,b=b,(a+2*b+i)%11\n    if b%2:\n        s+=a",[a,b,s])
db=sqlite3.connect(":memory:")
db.executescript("create table U(id); insert into U values (1),(2),(3),(4); create table E(uid,val); insert into E values (1,3),(1,5),(1,5),(2,2),(3,NULL),(3,7);")
query="SELECT COUNT(e.val) FROM U u LEFT JOIN E e ON e.uid=u.id AND e.val>=5 GROUP BY u.id ORDER BY u.id"
add("C3","SQLの外部結合","SQLの表U(id)の行は 1,2,3,4。表E(uid,val)は (1,3),(1,5),(1,5),(2,2),(3,NULL),(3,7)。重複行も含む。次の問い合わせが返す各行の数値を、返される順に答えよ。\n"+query,[k for k, in db.execute(query)])
queue=list(range(1,9)); killed=[]; i=0
while queue:
    i=(i+2)%len(queue); killed.append(queue.pop(i))
add("C4","循環キュー","1〜8がこの順で輪になっている。最初は1から数え、3番目を除く。除いた人の次の生存者を新たな1番目としてまた3番目を除く。1人になるまで繰り返す。[最初に除く番号, 4番目に除く番号, 最後に残る番号] を答えよ。",[killed[0],killed[3],killed[-1]])

spans=[(0,7),(3,10),(5,8),(10,14),(12,15)]
add("D1","区間の重なり","5台の機器の稼働区間（分）は [0,7),[3,10),[5,8),[10,14),[12,15)。開始端を含み終了端を含まない。同時にちょうど2台が稼働している時間を合計すると何分か。", sum(sum(a<=t<b for a,b in spans)==2 for t in range(15)))
events=[("a",1,8),("b",2,9),("a",1,20),("c",1,-3),("d",2,-4),("e",1,2),("b",2,-7)]
seen=set(); totals=[0,0]
for key,acct,v in events:
    if key not in seen:
        totals[acct-1]+=v; seen.add(key)
add("D2","重複イベントの集計","口座1と2の初期残高はともに0。イベントは (ID,口座,増減) の順に (a,1,+8),(b,2,+9),(a,1,+20),(c,1,-3),(d,2,-4),(e,1,+2),(b,2,-7)。同じIDは最初の1件のみ採用する。[口座1,口座2] の最終残高を答えよ。",totals)
values=[(6,2),(9,4),(4,3),(10,1),(7,5)]
data=sorted(v for v,c in values for _ in range(c))
add("D3","度数表の順位","値と度数の組は (6,2),(9,4),(4,3),(10,1),(7,5)。度数ぶん展開した全15個を小さい順に並べる。[第4番目, 第8番目, 第12番目] の値を答えよ。順位は1から数える。",[data[k-1] for k in (4,8,12)])
stock=7; lost=0
for demand,delivery in zip([9,2,8,4,5],[3,4,0,6,2]):
    lost+=max(demand-stock,0); stock=max(stock-demand,0)+delivery
add("D4","在庫の時系列","初日在庫は7。5日間の需要は順に9,2,8,4,5。各日の閉店後の入荷は順に3,4,0,6,2。毎日、需要を先に処理し、不足分は失注して翌日に持ち越さない。[5日目入荷後の在庫, 5日間の合計失注数] を答えよ。",[stock,lost])

add("P1","権限の共通部分","架空APIの仕様。項目は1から7。サーバーが読める項目は{1,2,4,5,7}、トークンが読める項目は{2,3,4,7}、組織が明示的に拒否した項目は{4,6}。両方の読み取り許可があり明示拒否のない項目だけ取得できる。取得できる番号を昇順に答えよ。",sorted(({1,2,4,5,7}&{2,3,4,7})-{4,6}))
add("P2","生データの所在","架空サービスの仕様。1=端末が氏名入り文章を作る。2=ゲートウェイがそれを平文で読み、氏名が復元不能な記号に置換する。3=推論サーバーは置換後の文章だけを読む。4=監査DBには処理時刻と件数だけが届く。5=鍵保管庫はAPIキーだけを返す。元の氏名を平文で扱う構成要素の番号をすべて昇順に答えよ。",[1,2])
times=[(1,1),(2,9),(3,23),(4,24),(5,36),(6,48)]
add("P3","保存期限の境界","架空ログの削除仕様。記録の年齢が24時間以上になった最初の定期削除で消える。定期削除は時刻0,12,24,36,48時間に動く。同時刻なら書き込みが削除より先。記録 (ID,書込時刻) は (1,1),(2,9),(3,23),(4,24),(5,36),(6,48)。時刻48の定期削除直後に残るIDを昇順に答えよ。",[i for i,t in times if 48-t<24])
requests=[(1,"社内","読","一般"),(2,"社外","読","一般"),(3,"社内","書","一般"),(4,"社内","書","機密"),(5,"社外","書","一般"),(6,"社内","読","機密")]
allowed=[i for i,net,op,kind in requests if (op=="読" or net=="社内") and not (kind=="機密" and op=="書")]
add("P4","許可と拒否の優先順","架空ポリシー。読み取りは許可。社内からの書き込みは許可。機密への書き込みは常に拒否。明示拒否が優先し、その他は既定で拒否。要求 (ID,接続元,操作,対象) は (1,社内,読,一般),(2,社外,読,一般),(3,社内,書,一般),(4,社内,書,機密),(5,社外,書,一般),(6,社内,読,機密)。許可されるIDを昇順に答えよ。",allowed)

def valid_path(steps):
    x=y=0
    for step in steps:
        x+=step=="E"; y+=step=="N"
        if (x,y) in {(2,1),(4,3)}: return False
    return True
walks=set(itertools.combinations(range(10),4))
add("M5","障害点のある格子路","格子上で (0,0) から (6,4) まで、右へ1か上へ1だけ動く最短経路を考える。(2,1) と (4,3) のどちらも通らない経路は何通りか。",sum(valid_path("N" if i in up else "E" for i in range(10)) for up in walks))
secret=(4,0,2)
guesses=[(0,1,2),(1,2,3),(3,4,0),(4,2,1),(0,2,5)]
def clue(code,guess):
    exact=sum(a==b for a,b in zip(code,guess))
    return exact,len(set(code)&set(guess))-exact
clues=[clue(secret,g) for g in guesses]
candidates=[p for p in itertools.permutations(range(6),3) if [clue(p,g) for g in guesses]==clues]
assert candidates==[secret]
examples="、".join("".join(map(str,g))+f"は({a},{b})" for g,(a,b) in zip(guesses,clues))
add("L5","暗証番号の制約充足","0〜5から重複なしで3個選んだ暗証番号。各予想の成績 (a,b) は位置まで一致した個数a、数字は含むが位置は違う個数b。"+examples+"。暗証番号を左から3個の整数の配列で答えよ。",list(candidates[0]))
def funcs():
    fs=[]
    for i in range(3):
        fs.append(lambda x,j=i: (x+i)*j)
    i=4
    return [f(k) for k,f in enumerate(fs,1)]
add("C5","クロージャの評価時点","Python 3で下の f() の戻り値は何か。\ndef f():\n    fs=[]\n    for i in range(3):\n        fs.append(lambda x,j=i: (x+i)*j)\n    i=4\n    return [fn(k) for k,fn in enumerate(fs,1)]",funcs())
groups={"A":[(1,2),(80,100)],"B":[(6,10),(9,10)]}
def rates(gs):
    return (sum((Fraction(a,b) for a,b in gs),Fraction())/2, Fraction(sum(a for a,b in gs),sum(b for a,b in gs)))
av,bv=map(rates,groups.values())
add("D5","群別平均と合算率","方式Aの2群の成功件数/試行件数は1/2と80/100。方式Bは6/10と9/10。比較①では各群の成功率を等重みで平均する。比較②では件数を合算した成功率を使う。各比較で成績の良い方法をA=1、B=2として [比較①,比較②] の順で答えよ。",[1 if a>b else 2 for a,b in zip(av,bv)])
tokens=[(1,4,12,"R"),(2,5,14,"W"),(3,5,11,"R"),(4,7,13,"R"),(5,8,14,"R"),(6,7,12,"R")]
add("P5","失効と権限の時系列","架空の認証仕様。時刻11の読み取りに使えるのは、権限Rで、有効期限が11より後、かつ発行時刻が5以上のトークン。さらにID4は個別に失効している。トークン (ID,発行時刻,有効期限,権限) は (1,4,12,R),(2,5,14,W),(3,5,11,R),(4,7,13,R),(5,8,14,R),(6,7,12,R)。使えるIDをすべて昇順に答えよ。",[i for i,issue,expiry,role in tokens if role=="R" and issue>=5 and expiry>11 and i!=4])
assert len(rows)==len({r["id"] for r in rows})==25
parser = argparse.ArgumentParser(description="Recompute the frozen questions and answer key.")
parser.add_argument("--output", type=Path, help="Write a rebuilt suite to this file")
args = parser.parse_args()
# Preserve the exact pre-run fixture bytes on all platforms.
data = json.dumps(rows, ensure_ascii=False, indent=2).replace("\n", "\r\n").encode("utf8")
if args.output:
    args.output.write_bytes(data)
elif data != Path(__file__).with_name("suite.json").read_bytes():
    raise ValueError("Recomputed questions or answers differ from suite.json")
else:
    print("Verified all 25 questions and expected answers.")
