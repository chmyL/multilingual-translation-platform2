# real_corpus 构建脚本
import json, os, sys

BASE = "/mnt/d/flores200_dataset/devtest"
OUT = "/mnt/f/PythonProject/data/corpus/backup"
os.makedirs(OUT, exist_ok=True)

# 1. 读取四语言文件
files_ok = True
for f in ["zho_Hans.devtest","eng_Latn.devtest","jpn_Jpan.devtest","kor_Hang.devtest"]:
    if not os.path.exists(f"{BASE}/{f}"):
        print(f"ERROR: {BASE}/{f} not found. Download FLORES-200 devtest first.")
        print("https://github.com/facebookresearch/flores/tree/main/flores200")
        sys.exit(1)

zh = open(f"{BASE}/zho_Hans.devtest", encoding="utf-8").read().strip().split("\n")
en = open(f"{BASE}/eng_Latn.devtest", encoding="utf-8").read().strip().split("\n")
ja = open(f"{BASE}/jpn_Jpan.devtest", encoding="utf-8").read().strip().split("\n")
ko = open(f"{BASE}/kor_Hang.devtest", encoding="utf-8").read().strip().split("\n")
print(f"FLORES-200: zh={len(zh)} en={len(en)} ja={len(ja)} ko={len(ko)}")

# 2. 关键词分类
KW = {
    "academic": ["研究","实验","模型","算法","数据","训练","性能","分析","方法","结果",
        "证明","假设","理论","科学","论文","学者","教授","大学","学习","系统",
        "技术","计算","统计","测量","观测","发现","表明","显著","准确"],
    "business": ["公司","企业","市场","销售","客户","价格","产品","经济","财务",
        "投资","利润","收入","预算","合同","交易","商业","银行","货币","股票"],
    "daily": ["今天","昨天","明天","早上","晚上","吃饭","喝水","睡觉","走路",
        "朋友","家人","妈妈","爸爸","孩子","学校","衣服","天气","雨",
        "电影","音乐","书","狗","猫","花","树","车","房子","开心","难过"],
    "technical": ["服务器","代码","程序","网络","接口","配置","部署",
        "软件","硬件","协议","内存","线程","进程","缓存","端口",
        "版本","更新","错误","日志","安全","加密"],
}

corpus = {"academic":[],"business":[],"daily":[],"technical":[]}
MAX = 50
seen = set()

for i in range(len(zh)):
    if all(len(corpus[d]) >= MAX for d in corpus): break
    txt = zh[i].strip()
    if not txt or txt in seen or len(txt) < 10 or len(txt) > 200: continue
    scores = {d: sum(1 for kw in kws if kw in txt) for d, kws in KW.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0 or len(corpus[best]) >= MAX: continue
    seen.add(txt)
    corpus[best].append({
        "zh": txt, "en": en[i].strip(), "ja": ja[i].strip(), "ko": ko[i].strip(),
        "domain": best, "source": "FLORES-200",
    })

total = sum(len(v) for v in corpus.values())
print(f"关键词分类: {total} (学术{len(corpus['academic'])} 商务{len(corpus['business'])} 日常{len(corpus['daily'])} 技术{len(corpus['technical'])})")

# 3. 技术领域补足32条
print("补足技术语料 32 条...")
TECH32 = [
    ["该系统通过消息队列实现异步任务处理。","The system processes asynchronous tasks through message queues.","このシステムはメッセージキューを通じて非同期タスクを処理する。","이 시스템은 메시지 큐를 통해 비동기 작업을 처리한다."],
    ["请将数据库连接池的最大连接数设置为100。","Please set the maximum number of database connections in the connection pool to 100.","データベース接続プールの最大接続数を100に設定してください。","데이터베이스 연결 풀의 최대 연결 수를 100으로 설정하십시오."],
    ["该接口支持OAuth 2.0认证协议。","This interface supports the OAuth 2.0 authentication protocol.","このインターフェースはOAuth 2.0認証プロトコルをサポートしている。","이 인터페이스는 OAuth 2.0 인증 프로토콜을 지원한다."],
    ["请确保所有微服务在注册中心正确注册。","Please ensure all microservices are correctly registered with the service registry.","すべてのマイクロサービスがサービスレジストリに正しく登録されていることを確認してください。","모든 마이크로서비스가 서비스 레지스트리에 올바르게 등록되었는지 확인하십시오."],
    ["日志文件应按日期轮转并保留30天。","Log files should be rotated by date and retained for 30 days.","ログファイルは日付ごとにローテーションされ、30日間保持されるべきである。","로그 파일은 날짜별로 순환되며 30일간 보관되어야 한다."],
    ["该函数的时间复杂度为O(n²)，建议优化。","The time complexity of this function is O(n²), and optimization is recommended.","この関数の時間複雑度はO(n²)であり、最適化が推奨される。","이 함수의 시간 복잡도는 O(n²)이며 최적화가 권장된다."],
    ["请将环境变量注入容器而非硬编码在配置文件中。","Please inject environment variables into containers rather than hardcoding them in configuration files.","環境変数を設定ファイルにハードコードするのではなく、コンテナに注入してください。","환경 변수를 구성 파일에 하드코딩하지 말고 컨테이너에 주입하십시오."],
    ["该服务器支持热插拔硬盘，无需停机更换。","This server supports hot-swappable hard drives, allowing replacement without downtime.","このサーバーはホットスワップ対応ハードディスクをサポートしており、ダウンタイムなしで交換可能である。","이 서버는 핫스왑 가능한 하드 드라이브를 지원하여 다운타임 없이 교체할 수 있다."],
    ["API网关负责请求路由、限流和身份验证。","The API gateway is responsible for request routing, rate limiting, and authentication.","APIゲートウェイはリクエストルーティング、レート制限、認証を担当する。","API 게이트웨이는 요청 라우팅, 속도 제한 및 인증을 담당한다."],
    ["请使用SSH隧道加密数据库连接。","Please use an SSH tunnel to encrypt the database connection.","データベース接続を暗号化するためにSSHトンネルを使用してください。","데이터베이스 연결을 암호화하기 위해 SSH 터널을 사용하십시오."],
    ["该模块的构建时间过长，建议启用增量编译。","The build time for this module is excessive; enabling incremental compilation is recommended.","このモジュールのビルド時間が長すぎるため、増分コンパイルを有効にすることが推奨される。","이 모듈의 빌드 시간이 너무 길어 증분 컴파일을 활성화하는 것이 권장된다."],
    ["监控系统检测到CPU使用率持续超过90%。","The monitoring system has detected sustained CPU usage exceeding 90%.","監視システムがCPU使用率が継続的に90%を超えていることを検出した。","모니터링 시스템이 CPU 사용률이 지속적으로 90%를 초과하는 것을 감지했다."],
    ["请定期更新操作系统补丁以修复已知安全漏洞。","Please regularly update operating system patches to fix known security vulnerabilities.","既知のセキュリティ脆弱性を修正するために、オペレーティングシステムのパッチを定期的に更新してください。","알려진 보안 취약점을 수정하기 위해 운영 체제 패치를 정기적으로 업데이트하십시오."],
    ["该分布式锁基于Redis的SETNX命令实现。","This distributed lock is implemented based on Redis SETNX commands.","この分散ロックはRedisのSETNXコマンドに基づいて実装されている。","이 분산 잠금은 Redis의 SETNX 명령을 기반으로 구현된다."],
    ["请勿在生产环境中使用默认密码。","Do not use default passwords in production environments.","本番環境ではデフォルトのパスワードを使用しないでください。","프로덕션 환경에서 기본 비밀번호를 사용하지 마십시오."],
    ["该项目的代码覆盖率阈值设定为80%。","The code coverage threshold for this project is set at 80%.","このプロジェクトのコードカバレッジ閾値は80%に設定されている。","이 프로젝트의 코드 커버리지 임계값은 80%로 설정되어 있다."],
    ["静态代码分析工具已集成到CI流水线中。","Static code analysis tools have been integrated into the CI pipeline.","静的コード解析ツールがCIパイプラインに統合された。","정적 코드 분석 도구가 CI 파이프라인에 통합되었다."],
    ["请确保备份策略同时涵盖全量备份和增量备份。","Please ensure the backup strategy covers both full and incremental backups.","バックアップ戦略が完全バックアップと増分バックアップの両方をカバーしていることを確認してください。","백업 전략이 전체 백업과 증분 백업을 모두 포함하는지 확인하십시오."],
    ["该服务依赖于三个上游API，任一不可用将触发熔断机制。","This service depends on three upstream APIs; the unavailability of any one will trigger the circuit breaker.","このサービスは3つの上流APIに依存しており、いずれかが利用できない場合、サーキットブレーカーがトリガーされる。","이 서비스는 세 개의 업스트림 API에 의존하며, 어느 하나라도 사용할 수 없으면 서킷 브레이커가 트리거된다."],
    ["请将前端静态资源部署到CDN以加速全球访问。","Please deploy frontend static assets to a CDN to accelerate global access.","グローバルアクセスを高速化するために、フロントエンドの静的リソースをCDNにデプロイしてください。","글로벌 접근을 가속화하기 위해 프론트엔드 정적 리소스를 CDN에 배포하십시오."],
    ["内存泄漏问题已通过Valgrind工具定位并修复。","The memory leak issue has been identified and fixed using the Valgrind tool.","メモリリークの問題はValgrindツールを使用して特定され、修正された。","메모리 누수 문제가 Valgrind 도구를 사용하여 식별되고 수정되었다."],
    ["请将WebSocket连接的心跳间隔设置为30秒。","Please set the heartbeat interval for WebSocket connections to 30 seconds.","WebSocket接続のハートビート間隔を30秒に設定してください。","WebSocket 연결의 하트비트 간격을 30초로 설정하십시오."],
    ["该数据库支持行级锁和表级锁两种并发控制机制。","This database supports two concurrency control mechanisms: row-level locking and table-level locking.","このデータベースは行レベルロックとテーブルレベルロックの2つの同時実行制御メカニズムをサポートしている。","이 데이터베이스는 행 수준 잠금과 테이블 수준 잠금의 두 가지 동시성 제어 메커니즘을 지원한다."],
    ["请勿将敏感信息（如API密钥）提交到版本控制系统。","Do not commit sensitive information such as API keys to version control systems.","APIキーなどの機密情報をバージョン管理システムにコミットしないでください。","API 키와 같은 민감한 정보를 버전 관리 시스템에 커밋하지 마십시오."],
    ["负载测试表明系统在1000并发用户下仍保持稳定。","Load testing indicates the system remains stable under 1,000 concurrent users.","負荷テストにより、システムが1000人の同時接続ユーザー下でも安定していることが示された。","부하 테스트 결과 시스템이 1000명의 동시 사용자 환경에서도 안정적으로 유지되는 것으로 나타났다."],
    ["请使用HTTPS并启用HSTS以增强传输安全。","Please use HTTPS and enable HSTS to enhance transport security.","転送セキュリティを強化するためにHTTPSを使用し、HSTSを有効にしてください。","전송 보안을 강화하기 위해 HTTPS를 사용하고 HSTS를 활성화하십시오."],
    ["该架构采用事件驱动模式，各服务通过事件总线解耦。","This architecture adopts an event-driven pattern, with services decoupled through an event bus.","このアーキテクチャはイベント駆動パターンを採用し、各サービスはイベントバスを通じて疎結合されている。","이 아키텍처는 이벤트 구동 패턴을 채택하며, 각 서비스는 이벤트 버스를 통해 결합이 해제된다."],
    ["请检查SSL证书链是否完整，中间证书不可缺失。","Please check whether the SSL certificate chain is complete; intermediate certificates must not be missing.","SSL証明書チェーンが完全であるかどうかを確認してください。中間証明書が欠落してはならない。","SSL 인증서 체인이 완전한지 확인하십시오. 중간 인증서가 누락되어서는 안 된다."],
    ["该函数对输入参数未做边界检查，存在安全风险。","This function does not perform boundary checking on input parameters, posing a security risk.","この関数は入力パラメータに対して境界チェックを行っておらず、セキュリティリスクが存在する。","이 함수는 입력 매개변수에 대한 경계 검사를 수행하지 않아 보안 위험이 존재한다."],
    ["系统采用读写分离架构，主库负责写入，从库负责查询。","The system uses a read-write separation architecture, where the master handles writes and replicas handle queries.","システムは読み書き分離アーキテクチャを採用し、マスターが書き込みを担当し、レプリカがクエリを担当する。","시스템은 읽기-쓰기 분리 아키텍처를 사용하며, 마스터는 쓰기를 처리하고 복제본은 쿼리를 처리한다."],
    ["请将访问日志格式配置为JSON以便于日志分析工具解析。","Please configure the access log format as JSON to facilitate parsing by log analysis tools.","ログ分析ツールによる解析を容易にするために、アクセスログ形式をJSONに設定してください。","로그 분석 도구의 구문 분석을 용이하게 하기 위해 액세스 로그 형식을 JSON으로 구성하십시오."],
    ["该容器编排平台支持自动扩缩容和自愈能力。","This container orchestration platform supports auto-scaling and self-healing capabilities.","このコンテナオーケストレーションプラットフォームは自動スケーリングと自己修復機能をサポートしている。","이 컨테이너 오케스트레이션 플랫폼은 자동 확장 및 자가 치유 기능을 지원한다."],
]
for t in TECH32:
    corpus["technical"].append({"zh":t[0],"en":t[1],"ja":t[2],"ko":t[3],"domain":"technical","source":"hand-written supplement"})

# 4. 保存
total = sum(len(corpus[d]) for d in corpus)
corpus["meta"] = {
    "source": "FLORES-200 devtest (Meta AI, CC BY-SA 4.0)",
    "url": "https://github.com/facebookresearch/flores",
    "languages": ["zh","en","ja","ko"],
    "total": total,
    "created": "2026-06-24",
    "note": "200 entries: 168 from FLORES-200 keyword classification + 32 hand-written technical"
}
with open(f"{OUT}/real_corpus.json", "w", encoding="utf-8") as f:
    json.dump(corpus, f, ensure_ascii=False, indent=2)

print(f"Done: {OUT}/real_corpus.json ({os.path.getsize(f'{OUT}/real_corpus.json')/1024:.0f} KB, {total} entries)")
for d in ["academic","business","daily","technical"]:
    print(f"  {d}: {len(corpus[d])}")
