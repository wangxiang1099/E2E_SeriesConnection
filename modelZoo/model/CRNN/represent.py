import torch
import torch.nn as nn
from tqdm import tqdm
import os

class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    keys = {
    'space': ' ',
    'chinese': u''';.:?!/、《》<>【】[]()°'"+-=*#abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789%$￥αβ的一是不在有人了中国大为上这个以年生和我时之也来到要会学对业出行公能他于而发地可作就自们后成家日者分多下其用方本得子高过经现说与前理工所力如将军部事进司场同机主都实天面市新动开关定还长此月道美心法最文等当第好然体全比股通性重三外但战相从你内无考因小资种合情去里化次入加间些度员意没产正表很队报已名海点目着应解那看数东位题利起二民提及明教问制期元游女并曰十果么注两专样信王平己金务使电网代手知计至常只展品更系科门特想西水做被北由万老向记政今据量保建物区管见安集或认程总少身先师球价空旅又求校强各非立受术基活反世何职导任取式试才结费把收联直规持赛社四山统投南原该院交达接头打设每别示则调处义权台感斯证言五议给决论她告广企格增让指研商客太息近城变技医件几书选周备流士京传放病华单话招路界药回再服什改育口张需治德复准马习真语难始际观完标共项容级即必类领未型案线运历首风视色尔整质参较云具布组办气造争往形份防它车深神称况推越英易且营条消命团确划精足儿局飞究功索走望却查武思兵识克故步影带乐白源史航志州限清光装节号转图根省许引势失候济显百击器象效仅爱官包供低演连夫快续支验阳男觉花死字创素半预音户约率声请票便构存食段远责拉房随断极销林亚隐超获升采算益优愿找按维态满尽令汉委八终训值负境练母热适江住列举景置黄听除读众响友助弹干孩边李六甚罗致施模料火像古眼搜离闻府章早照速录页卫青例石父状农排降千择评疗班购属革够环占养曾米略站胜核否独护钱红范另须余居虽毕攻族吃喜陈轻亲积星假县写刘财亿某括律酒策初批普片协售乃落留岁突双绝险季谓严村兴围依念苏底压破河怎细富切乎待室血帝君均络牌陆印层斗简讲买谈纪板希聘充归左测止笑差控担杀般朝监承播亦临银尼介博软欢害七良善移土课免射审健角伊欲似配既拿刚绩密织九编狐右龙异若登检继析款纳威微域齐久宣阿俄店康执露香额紧培激卡短群春仍伤韩楚缺洲版答修媒秦错欧园减急叫诉述钟遇港补送托夜兰诸呢席尚福奖党坐巴毛察奇孙竞宁申疑黑劳脑舰晚盘征波背访互败苦阶味跟沙湾岛挥礼词宝券虑徐患贵换矣戏艺侯顾副妇董坚含授皇付坛皆抗藏潜封础材停判吸轮守涨派彩哪笔氏尤逐冲询铁衣绍赵弟洋午奥昨雷耳谢乡追皮句刻油误宫巨架湖固痛楼杯套恐敢遂透薪婚困秀帮融鲁遗烈吗吴竟惊幅温臣鲜画拥罪呼警卷松甲牛诺庭休圣馆退莫讯渐熟肯冠谁乱朗怪夏危码跳卖签块盖束毒杨饮届序灵怀障永顺载倒姓丽靠概输货症避寻丰操针穿延敌悉召田稳典吧犯饭握染怕端央阴胡座著损借朋救库餐堂庆忽润迎亡肉静阅盛综木疾恶享妻厂杂刺秘僧幸扩裁佳趣智促弃伯吉宜剧野附距唐释草币骨弱俱顿散讨睡探郑频船虚途旧树掌遍予梦圳森泰慢牙盟挑键阵暴脱汇歌禁浪冷艇雅迷拜旦私您启纷哈订折累玉脚亮晋祖菜鱼醒谋姐填纸泽戒床努液咨塞遭玩津伦夺辑癌丹荣仪献符翻估乘诚川惠涉街诗曲孔娘怒扬闲蒙尊坦衡迪镇沉署妖脸净哥顶掉厚魏旗兄荐童剂乏倍萨偏洗惯灭径犹趋拍档罚纯洛毫梁雨瑞宗鼓辞洞秋郎舍蓝措篮贷佛坏俗殊炮厅筑姆译摄卒谷妈聚违忘鬼触丁羽贫刑岗庄伟兼乳叶凡龄宽峰宋硬岸迅喝拟雄役零舞暗潮绿倾详税酸徒伴诊跑吾燕澳啊塔宿恩忙督末伐篇敏贸巧截沟肝迹烟勇乌赞锋返迫凭虎朱拔援搞爆勤抢敬赶抱仁秒缓御唯缩尝贴奔跨炎汤侵骑励戴肤枪植瘤埃汽羊宾替幕贝刀映彻驻披抓奉抵肿麻炸繁赢茶伏梅狂忧豪暂贾洁绪刊忆桥晓册漫圆默妾侧址横偶狗陵伙杜忍薄雪陷仙恋焦焉烦甘腺颇赏肠废墙债艾杰残冒屋堡曹储莱挂纵孝珍麦逃奋览镜缘昭摆跌胁昌耶腹偿蛋盈瓦摩沈惟迁冰辛震旁泉圈巡罢泛穷伸曼滋丈颜勒悲肥郭混灯租鸡阻邑伍践驾魔拒懂糖脏沿翁胆惧聊携晨滑菌辅贤鉴丝尾赴吨宇眠脂籍彼污貌弄郡奶菲烧垂壮浮弗赖珠迟渠寿隆剑胞跃稍愈荷壁卿邦忠摇悟锦扰袭盾艘浓筹盗哭淡孕扣呈怨琳孤奴驱振闭隔寒汝贯恢饰荡姑械猛亏锁硕舒嘉宏劲帅誉番惜胸抽脉孟遣碍辆玄陶丧矿链矛鸟夷嘴坡吕侦鸣妹邓钢妙欣骗浙辽奏唱腐仆祝冬韦邮酬尺涯毁粉井腰肌搭恨乙勿婆闹猎厉哀递廉卧豆揭瓶蒋忌贡邀覆墓捷骂芳耗奈腾抑牵履绕睛炼描辉肃循仿葬漏恰殿遥尿凯仲婢胃翼卢慎厦颈哉疲惑汗衰剩昆耐疫霸赚彭狼洪枚媪纲窗偷鼻池磨尘账拼榜拨扫妆槽蔡扎叔辈泡伪邻锡仰寸盐叹囊幼拓郁桌舟丘棋裂扶逼熊轰允箱挺赤晶祭寄爷呆胶佩泪沃婴娱霍肾诱扁辩粗夕灾哲涂艰猪铜踏赫吹屈谐仔沪殷辄渡屏悦漂祸赔涛谨赐劝泌凤庙墨寺淘勃崇灰虫逆闪竹疼旨旋蒂悬紫慕贪慧腿赌捉疏卜漠堪廷氧牢吏帕棒纽荒屡戈氛黎桃幽尖猫捕嫁窃燃禽稿掩踪姻陪凉阔碰幻迈铺堆柔姿膜爸斤轨疆丢仓岂柳敦祥栏邪魂箭煤惨聪艳儒仇徽厌潘袖宅恒逻肺昂炒醉掘宪摸愤畅汪贺肪撑桂耀柏扑淮凌遵钻摘碎抛匹腔纠吐滚凝插鹰郊琴悄撤驶粮辱斩暖杭齿欺殖撞颁匈翔挤乔抚泥饱劣鞋肩雇驰莲岩酷玛赠斋辨泄姬拖湿滨鹏兽锐捧尸宰舆宠胎凶割虹俊糊兹瓜悔慰浦锻削唤戚撒冯丑亭寝嫌袁尉芬挖弥喊纤辟菩埋呀昏傅桑稀帐添塑赋扮芯喷夸抬旺襄岭颗柱欠逢鼎苗庸甜贼烂怜盲浅霞畏诛倡磁茨毅鲍骇峡妨雕袋裕哩怖阁函浩侍拳寡鸿眉穆狱牧拦雾猜顷昔慈朴疯苍渴慌绳闷陕宴辖舜讼柯丞姚崩绘枝牲涌虔姜擦桓逊汰斥颖悠恼灌梯捐挣衷啡娜旬呵刷帽岳豫咖飘臂寂粒募嘱蔬苹泣吊淳诞诈咸猴奸淫佐晰崔雍葛鼠爵奢仗涵淋挽敲沛蛇锅庞朵押鹿滩祠枕扭厘魅湘柴炉荆卓碗夹脆颠窥逾诘贿虞茫榻碑傲骄卑蓄煮劫卵碳痕攀搬拆谊禹窦绣叉爽肆羞爬泊腊愚牺胖弘秩娶妃柜躲葡浴兆滴衔燥斑挡笼徙憾垄肖溪叙茅膏甫缴姊逸淀擅催丛舌竭禅隶歧妥煌玻刃肚惩赂耻詹璃舱溃斜祀翰汁妄枭萄契骤醇泼咽拾廊犬筋扯狠挫钛扇蓬吞帆戎稽娃蜜庐盆胀乞堕趁吓框顽硅宛瘦剥睹烛晏巾狮辰茂裙匆霉杖杆糟畜躁愁缠糕峻贱辣歼慨亨芝惕娇渔冥咱栖浑禄帖巫喻毋泳饿尹穴沫串邹厕蒸滞铃寓萧弯窝杏冻愉逝诣溢嘛兮暮豹骚跪懒缝盒亩寇弊巢咬粹冤陌涕翠勾拘侨肢裸恭叛纹摊兑萝饥浸叟滥灿衍喘吁晒谱堵暑撰棉蔽屠讳庶巩钩丸诏朔瞬抹矢浆蜀洒耕虏诵陛绵尴坤尬搏钙饼枯灼饶杉盼蒲尧俘伞庚摧遮痴罕桶巷乖啦纺闯敛弓喉酿彪垃歇圾倦狭晕裤蜂垣莉谍俩妪钓逛椅砖烤熬悼倘鸭馈惹旭薛诀渗痒蛮罩渊踢崖粟唇辐愧玲遏昼芦纣琼椎咳熙钉剖歉坠誓啤碧郅吻莎屯吟臭谦刮掠垫宙冀栗壳崛瑟哄谏丙叩缪雌叠奠髃碘暨劭霜妓厨脾俯槛芒沸盯坊咒觅剪遽贩寨铸炭绑蹈抄阎窄冈侈匿斌沾壤哨僵坎舅洽勉侣屿啼侠枢膝谒砍厢昧嫂羡铭碱棺漆睐缚谭溶烹雀擎棍瞄裹曝傻旱坑驴弦贬龟塘贞氨盎掷胺焚黏乒耍讶纱蠢掀藤蕴邯瘾婿卸斧鄙冕苑耿腻躺矩蝶浏壶凸臧墅粘魄杞焰靶邵倚帘鞭僚酶靡虐阐韵迄樊畔钯菊亥嵌狄拱伺潭缆慑厮晃媚吵骃稷涅阪挨珊殆璞婉翟栋醋鹤椒囚瞒竖肴仕钦妒晴裔筛泻阙垒孰抖衬炫兢屑赦宵沮谎苟碌屁腕沦懈扉揖摔塌廖铝嘲胥曳敖傍筒朕扳鑫硝暇冶靖袍凑悍兔邢熏株哮鹅乾鄂矶逵坟佣髓隙惭轴掏苛偃榴赎谅裴缅皂淑噪阀咎揽绮瞻谜拐渭啥彦遁琐喧藉嫩寞梳溜粥恤迭瀑蓉寥彬俺忿螺膀惫扔匪毙怠彰啸荻逮删脊轩躬澡衫娥捆牡茎秉俭闺溺萍陋驳撼沽僮厥沧轿棘怡梭嗣凄铅绛祈斐箍爪琦惶刹嗜窜匠锤筵瑶幌捞敷酌阜哗聂絮阱膨坪歪旷翅揣樱甸颐兜頉伽绸拂狎颂谬昊皋嚷徊曙麟嚣哑灞钧挪奎肇磊蕉荧嗽瓒苯躯绎鸦茵澜搅渺恕矫讽匀畴坞谥趟蔓帛寅呜枣萌磷涤蚀疮浊煎叮倩拯瑰涩绅枉朽哺邱凿莽隋炳睁澄厄惰粤黯纬哦徘炜擒捏帷攒湛夙滤浐霄豁甄剔丫愕袜呕蹲皱勘辜唬葱甩诡猿稻宦姨橡涧亢芽濒蹄窍譬驿拢叱喂怯坝椰孽阖瞩萎镑簿婷咐郸瑜瑚矮祷窟藩牟疡仑谣侄沐孜劈枸妮蔚勋玫虾谴莹紊瓷魁淄扛曩柄滔缀闽莞恳磅耸灶埠嚼汲恍逗畸翩甥蚁耽稚戟戊侃帜璧碟敞晖匙烫眷娟卦寐苌馨锣谛桐钥琅赁蜡颤陇僻埔腥皎酝媳翘缔葫吼侮淹瘫窘啖犀弒蕾偕笃栽唾陀汾俨呐膳锌瞧骏笨琢踩濮黛墟蒿歹绰捍诫漓篷咄诬乓梨奕睿嫡幢砸俞亟捣溯饵嘘砂凰丕荥赀薇滕袱辍疹泗韧撕磕梗挚挠嫉奚弩蝉罐敝鞍晦酣搁柿菠卞煞堤蟹骼晤潇胰酱郦脖檐桩踵禾狩盏弈牒拙喇舶炊喀黔挟钞缕俏娄粪颅锏凹饲肘赟吝襟琪谕飙秽颊渝卯捡氢桀裳滇浇礁蚊芙荀吩凳峨巍雉郢铲倪杳汹豚乍蛙驼嗅讫痰棵睫绒捻罔杠氟堰羁穰钠骸睾鳞邸於谧睢泾芹钾颓笋橘卉岐懿巅垮嵩柰鲨涡弧钝啃熹芭隅拌锥抒焕漳鸽烘瞪箕驯恃靴刁聋剿筝绞鞅夯抉嘻弛垢衾丐斟恙雁匮娼鞠扼镶樵菇兖夭戌褚渲硫挞衙闫绾衅掣磋袒龚叨揉贻瑛俾薯憎傣炬荤烁沂粑蚌渣茄荼愍蒜菱狡蠡戍畤闵颍酋芮渎霆哼韬荫辙榄骆锂肛菑揪皖秃拽诟槐髦脓殡闾怅雯戮澎悖嗓贮炙跋玮霖皓煽娠肋闸眩慷迂酉蝇羌蔑氯蚕汀憋臾汕缸棚唉棕裟蚡驮簇橙蹇庇佼禧崎痘芜姥绷惮雏恬庵瞎臀胚嘶铀靳呻膺醛憧嫦橄褐讷趾讹鹊谯喋篡郝嗟琉逞袈虢穗踰栓钊羹掖笞恺掬憨狸瑕匡痪冢梧眺佑愣撇阏疚攘昕瓣烯谗隘酰绊鳌俟嫔崭妊荔毯纶祟爹辗竿裘犁柬恣阑榆翦佟钜札隧腌砌酥辕铬痔讥毓橐跻酮殉哙亵锯糜壬瞭恻轲糙涿绚荟梢赣沼腑朦徇咋膊陡骋伶涓芷弋枫觑巳匣蠕恪槟栎噩葵殃淤诠昵眸馁奄绽闱蛛矜馔遐骡罹遑隍拭祁霁釜钵栾睦蚤咏憬韶圭觇芸氓伎氮靓淆绢眈掐簪搀玺镐竺峪冉拴忡卤撮胧邛彝楠缭棠腮祛棱睨嫖圉杵萃沁嬉擂澈麽轸彘褥廓狙笛彗啬盂贲忏驺悚豨旌娩扃蹦扈凛驹剃孺吆驷迸毗熔逍癸稼溥嫣瓮胱痊疟拣戛臻缉懊竣囤侑肽缮绥踝壑娴猝焻禀漱碁蹬祗濡挝亳萦癖毡锈憩筷噬珀砝鬓瑾澧栈搓褒疤沌镖塾钗骊拷铂窒驸裨矗烙惬炖赍迥蹴炽诧闰糯捅茜漯峭哇鹑疵梓骠咫鹦檀痹侥蘑衢灸琵琶懦邺扪痿苔拇腋薨馅敕捂栅瓯嘿溉胳拎巿赃咕诃谤舁禺榨拈瘙眯篱鬟咯抨桨岱赡蹶惚嗔喏聆曜窑瘢柠蕃寤攫饷佬臼皈蟒啜蔗汶酪豕窖膛檬戾蟠黍鲸漾猾驭踊稠脯潍倭谑猖聒骞熄渍瞳蒯褪筐彤嬴沱闼橱蜚蹭臆邳盔眶沓飨覃彷淌岚霹袂嗤榔鸾綦莘媲翊雳蚩茸嗦楷韭簸帚坍後璋剽渤骥犊迩悯饪搂鹉岑觞棣蕊诳黥藻郜舵毂茗忱铿谙怆钳佗瀚亘铎咀濯鼾酵酯麾笙缨翳龈忒煦顼俎圃刍喙羲陨嘤梏颛蜒啮镁辇葆蔺筮溅佚匾暄谀媵纫砀悸啪迢瞽莓瞰俸珑骜穹麓潢妞铢忻铤劾樟俐煲粱虱徼脐嘈悴捶嚏挛谚螃殴瘟掺酚梵栩褂摹蜿钮箧胫馒焱嘟芋踌圜衿峙宓腆佞砺婪瀛苷昱贰秤扒躇翡宥弼缤鳖擞眨礶锢辫儋纭洼漕飓纂舷勺诲捺瞑啻蹙佯茹怏蛟鹭烬兀檄浒胤踞僖卬璀暧蚂饽镰陂瞌诽钺沥镍耘燎祚莺屎辘鸥氐匕銮苴憔渥袅瞿瓢痣蘸蹑玷惺轧喃潺唏逅懵帏唠徨咤抠蛊苇铮疙闳砥羸遨哎捽钏壹昇擢贽汴砰牝蔼熠粽绌杼麒叭颔锭妍姒邂轶搔蹊阂垦猕伫瘩璐黠婺噫潞呱幡汞缯骁墩瞥媛瞠羔轼拗鹞诮趴凋撩芥缎摒泮惘骛瘳姝渚吠稣罄吒茧黜缢獗诅絜蜕屹哽缄俑坷杓剁锺鹜谩岔籽磬溍邃钨甬蝠龋鸱孚馍溴妫偎烽椽阮酗惋牍觥瞅涣狈锰椟饺溲谪掇倔猢笄翕嗥狞洮炕瘠磺肱奭耆棂娅咚豌樗诩斡榈琛狲蕲捎戳炯峦嘎睬怙疱霎哂鱿涸咦痉抟庖沅瑙珏祜楞漉鸠镂诰谄蜗嗒珂祯鸳殒潼柩萤柑缰淼冗蕙鳄嘀彊峥雹藜笠岖傥潦苞蛰僦碣疸湮昴榷涎攸砾跖恂麝貂孢捋笈璨粕浚鹃歆漪岷咧殁篆湃侏傈殇霭嚎拊崂鬲碉菁庾旃幺皿焊噢祺锚痤翎醺噶傀俛秧谆绯瘥盥蹋髯岌痧偌禳簧跤伉腼爰箫曦蜘霓愆姗陬楂嵘蜓浼癫瓠跷绐枷墀馕盹聩镯砚晁坂煜俚眛焘阍袄馋泸庠毐飚刭琏羿斓稔阉喾恸耦咪蝎唿桔缑诋訾迨鹄蟾鬣廿莅荞槌媾愦郏淖嗪镀畦颦浃牖襁怂唆嚭涟拮腓缥郫遴嗝跛掂撬鄣鄱斫窿兕壕疽铙吱厩甭篝踣眦啧糠鲤粲噱椭哟潸铆姣馥胙迦偻嗯陟桧鸯恿晌骈喽淅澹叽桢刨忑忐猩蝙旄晾吭荏觐胄榛豢堑帔咙柚僭锵肮囿忤惴燮棹摈缈幛墉诎仞剌氇泯茱獾豺蜃殂窈倨褓詈砷邕薰焖獐雎帧鸩匝桅椁绫桡氆哌咛鞘辎缙玑佤垓槿蛤烨泓罴鄜褶瘀颌蹂弑珪曷膑惦咆梆蛾牂髅捱拧婧踱怵侗屉讪衲麋宕畿唧怛豉籁舂蓦廨胪怍鄄绶飕蜻欷汧唑冽邰魇铐哝泱扞飒醴陲喟筠殓瘸倏啕睑翌幄娓妩奁璜桦朐榕礴儡婕觎觊绦猥涮倬袤啄掳椿俪噜摞鄗漩悝淞袴僇酹搒鳍疣姁猗舛鞮砭郯徕纥梃卮肣湎怦揄迕芍珥羚喔缁涝栉犷汜悻呛赭淬泫炀箴镌髫拄怔炷桎巽汭挈噙锄邴歔瘪腴呗慵撺欤阡傩苫掰盅冑躏茉霾耄楹苋鲠哆傒榭牦婶仃囱皙暲砒舀鹗犒斛甑楫嫪胭瘁铛藕腭睽阕裀砧蓼搽荃奘祎泵攥翱晟酎逋箔羟诙饬跆眇佻铠娑郧葭蝗碾硒釉磔殄藐莠颧熨獠浞笺癣茬衽喳裾倜鸢蠹惆芈燔伛妗佃缜咣龛挎徵锉啾隼猬镳璇胯饕揩虮苓噎祓筰奂搪喁俦隗馏圩褫吮哧湫旻筏佶茕铣娆揍嗷柈蕨绖旎汨畑厝楯祇怼焯柘骷澍珞殚瑁蓐蹿犴孵筱蜷窋泞肄祐窕酆阗镝匍腱仡樾驽峒蟆徉昙罡耜嗨氲骅襦浔纮洱氦舐黙臊汛蹀溟枥祉铄豸揶馀呷仄焒嗡崆皑匐诿鲲筴侬鹳滂橹邈弭弁樽幔纨帼氤旒旖屣孱槁沣娣壅枇讴阆杷浣狰愠蚓咿藿萸刽稞刎骖骰嵯濂跚湄珰舔谮坨锲煨绻楣谟嗖裆晗囹黝讣貉椹蜇箩妤搐呦恽赊侩猱遒鸮迤凫诂骀瘴螨臃葩篓谲悌嬗颉赉珈汩薮鬃噤湍畲徜衮茀蓍遛磐蹒鸵褴苒郈踽叵伋襆伧茴赳矾圄楮坯蕤迓锱腉滦饯诤呤纡隽妲噻愀龊镭藓镣滈蓓杪糗菅椀懑苎劓囫啰钼烷兒脔郴忖芎啶巉钒缒蝼龌沔晔孳嗫宸佰蜈酞蔷糅猊缟郐眙赅剜徭蛭愎唔瘘镉殛茏邋垛垩焙羯浍鏖嚓烩莴绠纔衩糁町粝穑葺徂棓泷涪囵屦裱缱圹罂荦腈遢曛粳舫窭濠跄琥竽膈荚笮嶙靛虬赝篑侪矽堙泠瞟癀酤涔唁郿爻盱菡绨醚岿椐蔟螂辂窠淙铳孥蚣唳纻甾膘脍赈魉纰岫坌捭睒轺锗縻荼嗑瞋绔喱痞咔埤疥猷洺啁讦餮泅蛹癞琮铨杌孑菟骐峇缶茭煅酩酢苣蛆鸨愫骧茁黟荠饴绡酊蛀娲娉抡睚跹榫沬崴颚嫚珮谇瓴疴偈诒讧赪徳滁孪秸叼硼楸烝炔耙腩樯噔碴佝峤峣汐呓狒坻趵剎啐嘭噌噗圯坳柢踹镊呃幂蛐凇璟遹肓剐垝杅撷佘蚝栀枵蟋嗌玦唢喹珅喆谔钎讵钰脁柞叁柒捌玖仟掸秆埂硷铰撅谰沤呸沏扦瓤蓑誊烃铡诌俶''',
    'english': u'''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ''',
    'number': u'''0123456789''',
    'symbol': u'''.:?/、《》>【】[()°+-=*#''',
    'others': u'''''',
    'VAT': u''' .:/、《》<>【】[]()°+-=*#abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789%￥αβ的一是不在有人了中国大为上这个以年生和我时之也来到要会学对业出行公能他于而发地可作就自们后成家日者分多下其用方本得子高过经现说与前理工所力如将军部事进司场同机主都实天面市新动开关定还长此月道美心法最文等当第好然体全比股通性重三外但战相从你内无考因小资种合情去里化次入加间些度员意没产正表很队报已名海点目着应解那看数俶鹄东位题利起二民提及明教问制期元游女并十果么注两专样信王平己金务使电网代手知计至常只展品更系科门特想西水做被北由万老向记政今据量保建物区管见安集或认程总少身先师球价空旅又求校强各非立受术基活反世何职导任取式试才结费把收联直规持赛社四山统投南原该院交达接头打设每别示则调处义权台感斯证言五议给决论她告广企格增让指研商客太息近城变技医件几书选周备流士京传放病华单话招路界药回再服什改育口张需治德复准马习真语难始际观完标共项容级即必类领未型案线运历首风视色尔整质参较云具布组办气造争往形份防它车深神称况推越英易且营条消命团确划精足儿局飞究功索走望却查武思兵识克故步影带乐白源史航志州限清光装节号转图根省许引势失候济显百击器象效仅爱官包供低演连夫快续支验阳男觉花死字创素半预音户约率声请票便构存食段远责拉房随断极销林亚隐超获升采算益优愿找按维态满尽令汉委八终训值负境练母热适江住列举景置黄听除读众响友助弹干孩边李六甚罗致施模料火像古眼搜离闻府章早照速录页卫青例石父状农排降千择评疗班购属革够环占养曾米略站胜核否独护钱红范另须余居虽毕攻族吃喜陈轻亲积星假县写刘财亿某括律酒策初批普片协售乃落留岁突双绝险季谓严村兴围依念苏底压破河怎细富切乎待室血帝君均络牌陆印层斗简讲买谈纪板希聘充归左测止笑差控担杀般朝监承播亦临银尼介博软欢害七良善移土课免射审健角伊欲似配既拿刚绩密织九编狐右龙异若登检继析款纳威微域齐久宣阿俄叁柒捌玖仟店康执露香额紧培激卡短群春仍伤韩楚缺洲版答修媒秦错欧园减急叫诉述钟遇港补送托夜兰诸呢席尚福奖党坐巴毛察奇孙竞宁申疑黑劳脑舰晚盘征波背访互败苦阶味跟沙湾岛挥礼词宝券虑徐患贵换矣戏艺侯顾副妇董坚含授皇付坛皆抗藏潜封础材停判吸轮守涨派彩哪笔氏尤逐冲询铁衣绍赵弟洋午奥昨雷耳谢乡追皮句刻油误宫巨架湖固痛楼杯套恐敢遂透薪婚困秀帮融鲁遗烈吗吴竟惊幅温臣鲜画拥罪呼警卷松甲牛诺庭休圣馆退莫讯渐熟肯冠谁乱朗怪夏危码跳卖签块盖束毒杨饮届序灵怀障永顺载倒姓丽靠概输货症避寻丰操针穿延敌悉召田稳典吧犯饭握染怕端央阴胡座著损借朋救库餐堂庆忽润迎亡肉静阅盛综木疾恶享妻厂杂刺秘僧幸扩裁佳趣智促弃伯吉宜剧野附距唐释草币骨弱俱顿散讨睡探郑频船虚途旧树掌遍予梦圳森泰慢牙盟挑键阵暴脱汇歌禁浪冷艇雅迷拜旦私您启纷哈订折累玉脚亮晋祖菜鱼醒谋姐填纸泽戒床努液咨塞遭玩津伦夺辑癌丹荣仪献符翻估乘诚川惠涉街诗曲孔娘怒扬闲蒙尊坦衡迪镇沉署妖脸净哥顶掉厚魏旗兄荐童剂乏倍萨偏洗惯灭径犹趋拍档罚纯洛毫梁雨瑞宗鼓辞洞秋郎舍蓝措篮贷佛坏俗殊炮厅筑姆译摄卒谷妈聚违忘鬼触丁羽贫刑岗庄伟兼乳叶凡龄宽峰宋硬岸迅喝拟雄役零舞暗潮绿倾详税酸徒伴诊跑吾燕澳啊塔宿恩忙督末伐篇敏贸巧截沟肝迹烟勇乌赞锋返迫凭虎朱拔援搞爆勤抢敬赶抱仁秒缓御唯缩尝贴奔跨炎汤侵骑励戴肤枪植瘤埃汽羊宾替幕贝刀映彻驻披抓奉抵肿麻炸繁赢茶伏梅狂忧豪暂贾洁绪刊忆桥晓册漫圆默侧址横偶狗陵伙杜忍薄雪陷仙恋焦焉烦甘腺颇赏肠废墙债艾杰残冒屋堡曹储莱挂纵孝珍麦奋览镜缘昭摆跌胁昌耶腹偿蛋盈瓦摩沈惟迁冰辛震旁泉圈巡罢泛穷伸曼滋丈颜勒悲肥郭混灯租鸡阻邑伍践驾魔拒懂糖脏沿翁胆惧聊携晨滑菌辅贤鉴丝尾赴吨宇眠脂籍彼污貌弄郡奶菲烧垂壮浮弗赖珠迟渠寿隆剑胞跃稍愈荷壁卿邦忠摇悟锦扰袭盾艘浓筹盗哭淡孕扣呈怨琳孤奴驱振闭隔寒汝贯恢饰荡姑械猛亏锁硕舒嘉宏劲帅誉番惜胸抽脉孟遣碍辆玄陶丧矿链矛鸟夷嘴坡吕侦鸣妹邓钢妙欣骗浙辽奏唱腐仆祝冬韦邮酬尺涯毁粉井腰肌搭恨乙勿婆闹猎厉哀递廉卧豆揭瓶蒋忌贡邀覆墓捷骂芳耗奈腾抑牵履绕睛炼描辉肃循仿葬漏恰殿遥尿凯仲婢胃翼卢慎厦颈哉疲惑汗衰剩昆耐疫霸赚彭狼洪枚媪纲窗偷鼻池磨尘账拼榜拨扫妆槽蔡扎叔辈泡伪邻锡仰寸盐叹囊幼拓郁桌舟丘棋裂扶逼熊轰允箱挺赤晶祭寄爷呆胶佩泪沃婴娱霍肾诱扁辩粗夕灾哲涂艰猪铜踏赫吹屈谐仔沪殷辄渡屏悦漂祸赔涛谨赐劝泌凤庙墨寺淘勃崇灰虫逆闪竹疼旨旋蒂悬紫慕贪慧腿赌捉疏卜漠堪廷氧牢吏帕棒纽荒屡戈氛黎桃幽尖猫捕嫁窃燃禽稿掩踪姻陪凉阔碰幻迈铺堆柔姿膜爸斤轨疆丢仓岂柳敦祥栏邪魂箭煤惨聪艳儒仇徽厌潘袖宅恒逻肺昂炒醉掘宪摸愤畅汪贺肪撑桂耀柏扑淮凌遵钻摘碎抛匹腔纠吐滚凝插鹰郊琴悄撤驶粮辱斩暖杭齿欺殖撞颁匈翔挤乔抚泥饱劣鞋肩雇驰莲岩酷玛赠斋辨泄姬拖湿滨鹏兽锐捧尸宰舆宠胎凶割虹俊糊兹瓜悔慰浦锻削唤戚撒冯丑亭寝嫌袁尉芬挖弥喊纤辟菩埋呀昏傅桑稀帐添塑赋扮芯喷夸抬旺襄岭颗柱欠逢鼎苗庸甜贼烂怜盲浅霞畏诛倡磁茨毅鲍骇峡妨雕袋裕哩怖阁函浩侍拳寡鸿眉穆狱牧拦雾猜顷昔慈朴疯苍渴慌绳闷陕宴辖舜讼柯丞姚崩绘枝牲涌虔姜擦桓逊汰斥颖悠恼灌梯捐挣衷娜刷帽岳豫咖飘臂寂粒募嘱蔬苹泣吊淳诞诈咸猴佐晰崔葛鼠爵奢仗涵淋沛蛇锅庞朵鹿滩祠枕厘魅湘柴炉荆卓碗夹脆颠窥逾虞茫榻傲骄蓄煮碳痕攀禹窦绣叉泊腊愚牺胖弘秩娶妃柜躲葡浴兆滴衔燥斑挡笼徙憾垄肖溪叙茅膏甫缴姊逸淀擅催丛舌竭禅隶歧妥煌玻刃肚惩赂耻詹璃舱溃斜翰汁枭萄契醇拾廊犬筋扯狠挫钛扇蓬帆戎稽娃蜜庐盆胀框顽硅宛瘦睹烛晏巾狮辰茂裙霉杖杆糟畜糕峻辣亨芝娇渔冥咱栖浑禄帖巫喻毋泳尹穴沫串邹厕铃寓萧弯窝杏冻愉诣溢兮暮豹懒缝盒寇巢陌涕翠勾侨肢纹摊兑萝灿衍谱暑棉屠巩钩丸诏朔矢浆蜀洒耕虏诵陛绵坤搏钙饼枯灼饶杉盼蒲尧伞庚痴罕桶巷乖纺敛弓喉酿彪垃歇圾倦狭晕裤蜂垣莉谍俩钓逛椅砖烤熬悼倘鸭馈惹旭薛诀渗痒蛮罩渊崖粟唇辐玲遏昼芦纣琼椎咳熙钉剖坠啤碧郅莎屯臭谦刮垫宙冀栗壳瑟丙缪雌叠碘暨劭霜厨脾俯槛芒沸坊咒觅剪贩寨铸炭蹈阎窄冈侈斌壤哨僵坎舅洽侣屿侠枢膝谒厢昧嫂羡铭碱漆谭雀擎棍驴弦龟塘贞氨胺乒讶纱掀蕴邯斧冕苑耿矩蝶浏壶凸墅粘魄杞焰邵倚帘僚酶靡阐韵迄樊畔钯菊亥嵌狄伺潭缆慑厮媚吵骃稷阪珊殆璞婉翟栋醋鹤椒竖肴仕钦妒晴裔阙垒孰衬炫兢屑宵苟碌屁腕沦懈扉塌廖铝胥曳敖傍筒朕扳鑫硝暇冶靖袍凑悍兔邢熏株哮鹅乾鄂矶逵佣髓隙惭轴掏苛偃榴谅裴缅皂淑噪阀咎绮瞻谜渭彦藉嫩寞梳溜粥恤迭瀑蓉寥彬俺忿螺膀惫怠彰啸荻删脊轩躬衫娥牡茎秉俭闺萍陋沽僮沧轿棘怡梭嗣凄铅绛祈斐箍爪琦惶刹匠锤筵瑶幌捞敷酌阜聂絮阱膨坪歪旷翅樱甸颐兜頉伽绸拂颂谬昊皋嚷徊曙麟嚣哑灞钧挪奎肇磊蕉荧嗽瓒苯躯绎鸦茵澜渺矫匀畴坞谥趟蔓帛寅枣萌磷涤蚀疮浊煎叮倩瑰涩绅枉朽邱凿莽隋炳澄厄惰粤黯纬徘炜帷湛滤浐霄豁甄剔丫愕袜蹲皱勘辜葱甩猿稻橡涧亢芽濒蹄驿怯坝椰孽阖瞩萎镑簿婷郸瑜瑚矮祷窟牟疡仑谣侄沐孜劈枸妮蔚勋玫虾谴莹紊瓷魁淄柄滔缀闽莞恳磅耸灶埠嚼汲逗畸翩甥蚁耽稚戟帜璧碟敞晖匙烫眷娟卦寐苌馨锣谛桐钥琅赁蜡陇僻埔腥皎酝媳翘缔葫吼侮犀蕾偕笃栽唾陀汾俨膳锌瞧骏笨琢踩濮黛墟蒿歹绰漓篷咄诬乓梨奕睿嫡幢砸俞亟捣溯饵砂凰丕荥薇滕袱辍疹泗韧撕磕梗挚挠嫉奚弩蝉罐鞍晦酣柿菠卞煞堤蟹骼潇胰酱郦脖檐桩禾弈牒拙喇舶炊黔钞缕俏颅锏凹饲肘赟吝襟琪谕秽颊渝卯氢滇礁蚊芙荀吩凳峨巍雉郢铲倪杳汹豚乍蛙驼嗅痰棵绒杠氟谧芹笋橘卉岐懿巅嵩鲨熹芭隅拌锥抒焕漳鸽驯恃靴刁聋筝绞弛衾丐恙雁匮鞠镶菇兖夭戌褚渲硫挞闫绾龚贻瑛薯傣炬荤烁蚌渣茄蒜菱闵颍酋芮霆韬荫辙榄骆锂肛皖秃槐髦脓殡闾怅雯戮澎悖嗓贮炙跋玮霖皓煽娠肋闸眩慷迂酉蝇蔑氯蚕汀汕缸棚唉棕橙禧崎痘芜姥绷雏恬庵瞎胚铀靳醛憧嫦橄褐趾鹊郝琉穗踰栓钊羹掖笞恺掬憨狸瑕匡痪冢梧眺佑愣撇疚攘瓣谗隘绊鳌嫔崭妊荔毯纶祟爹辗竿裘榆佟钜札隧腌砌酥铬痔毓跻酮锯糜壬轲糙绚荟梢赣沼腑朦徇咋膊骋伶涓芷弋枫匣恪槟栎葵昵眸馁奄绽闱遑隍拭祁霁釜钵栾咏憬韶圭觇芸氮靓绢眈簪搀玺镐竺峪冉卤胧彝楠缭棠腮祛棱杵萃沁嬉擂澈褥廓狙笛彗盂忏娩蹦驹剃孺熔逍瓮胱疟拣戛臻囤娴漱蹬祗濡挝亳萦癖毡锈筷噬珀砝瑾栈搓褒塾钗骊拷铂矗烙蹴炽诧闰糯捅茜峭哇鹑梓骠鹦檀蘑灸琵琶懦馅巿赃咕诃禺榨拈瘙眯篱抨桨岱赡蹶聆曜窑柠饷佬皈蟒蔗汶酪豕窖檬蟠黍鲸漾猾踊稠脯潍骞熄渍瞳褪筐彤橱蜚蹭臆邳盔眶沓覃彷淌岚嗤榔鸾莘媲翊雳蚩茸楷韭簸帚坍璋渤悯饪搂藻茗瀚亘铎鼾酵酯麾笙缨龈煦羲陨嘤梏颛镁辇葆佚匾暄谀纫砀迢莓瞰俸珑骜穹麓潢妞铢忻铤劾樟俐煲粱挛谚螃殴瘟掺酚梵栩褂摹蜿馒焱嘟芋峙宓腆苷昱贰秤翡宥缤鳖眨锢辫儋漕勺佯茹怏蛟鹭烬兀檄浒璀暧蚂饽诽沥镍耘燎祚莺屎辘鸥氐匕銮苴憔渥袅瞿轧逅帏苇铮羸遨壹汴蔼粽杼麒锭妍姒轶猕璐婺潞幡汞羔轼撩芥缎姝渚稣罄吒茧黜诅絜蜕屹哽俑坷杓剁锺鹜籽锰饺倔猢炕瘠磺娅咚豌诩榈琛捎戳炯峦疱霎鱿痉庖沅瑙珏祜鸠镂蜗嗒珂祯鸳潼柩萤柑缰淼冗蕙鳄彊峥雹笠岖傥潦苞蛰僦碣湮昴榷涎攸砾跖恂麝貂孢捋笈璨粕浚鹃歆篆湃侏崂碉菁旃皿焊锚痤翎醺傀俛秧谆绯瘥盥蹋髯岌痧偌簧跤伉腼箫曦蜘霓姗楂嵘蜓癫跷绐枷馕砚坂桔莅荞漩悝淞袴僇酹搒鳍疣猗舛鞮砭纥卮怦揄迕芍珥羚喔缁阡茉霾榭牦婶仃囱皙暲砒舀鹗犒斛楫胭瘁铛藕腭睽阕葭硒惆芈佃缜挎徵锉啾隼猬璇胯饕揩苓噎奂搪馏圩吮哧湫筏茕铣娆澍珞殚瑁蹿犴孵筱蜷窋泞肄祐窕阗匍腱驽峒蟆徉昙罡氲骅襦浔纮洱氦舐汛蹀溟枥祉铄揶馀崆皑匐诿鲲侬鹳滂橹弭樽幔纨帼孱沣娣枇杷浣愠藿煨楣谟晗黝讣貉蜇箩妤呦恽赊茴矾锱饯隽妲蓓糗苎劓啰钼烷钒宸佰蜈酞蔷郐赅垛垩焙羯浍烩莴衩葺泷涪囵屦裱缱圹罂荦腈粳舫濠竽膈荚笮嶙靛虬赝篑侪矽泠瞟酤唁盱菡醚岿蔟螂淙榫沬徳孪秸耙骐肆骤趁蒸逝浸堵溶烹卸腻拱囚揽逮揣搁狩钠弧衙缮窒栅膛圃墩孚惋楞镯镀淅囿薰槿烨泵晟湍镭桕藩绑橇榉嵊槭龛晃裹缚婵塍沂菏衢漯郴娄碚綦莆坻蓟泸堰绥淖犁喀灏珩珲骥婕婧琨湄勉珉祺芊榕蕊爽昕萱滢雍钰煜媛霏烽桦珈姣珺藜咪崴葳煊昶闯恭琥颉筠淇焘藤琰旸邺谊玥昀喆嫒翱粲椿琮棣璠颢璟侃忱绫芩蕤裳菘菀琬骁岫瑄晔羿熠吟璎瀛彧桢斓聃淦漪拯泓聿舸庶馥峤焯铧侗昉圻沆璘铠锴荃苒嫣僖炀怿翌邕於鸯瑗韫茁涣璜骧猷佼泾竣镛勐颀恕洼燮琇铉珣旖瑀飙寰旌揆旎娉濂洵祎樵夔捍晁胄霭桉贲镔钿谌甬蕃珙炅粹绶玑楹墉稼戬玠垦铿鹂鲤夙俪涟琏鎏嫚睦钮溥嫱琚汐悌飒邈嵬娓遐滟榛锵臧箴诤洙倬琤囡迦蠡膺赜邸玎讷沄蔺郜墀旬艮勖磬玢磐砺拴坑阮隗晞泱扈骝驷郗嵇缙阚蒯邝诠邰邬冼铖訾''',
    'VAT-engnum': u'''0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ''',
    'SYN80k':u''' `|\~!@#$%^&*()_,;.:?/、《》<>【】{[]()}°'"+-=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789%$￥αβ'''
    }


    def __init__(self, alphabet_name, ignore_case=False):

        alphabet = self.keys[alphabet_name]
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def __len__(self):
        return len(self.dict)

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        length = []
        result = []
        decode_flag = True if type(text[0])==bytes else False

        for item in text:

            if decode_flag:
                item = item.decode('utf-8','strict')
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                result.append(index)
                
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return [''.join(char_list)]
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.extend(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


if __name__ == "__main__":

    c = strLabelConverter('VAT') 
    r = c.encode(["aaa",'123'])
    t = c.decode(torch.IntTensor([22,22,22,75,76,77]), torch.IntTensor([3,3]))
    print(t)
