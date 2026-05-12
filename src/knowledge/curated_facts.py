"""Curated facts about Cloud.ru capabilities — fallback for empty RAG.

Когда Managed RAG не возвращает чанков по теме (например, OBS WORM, PFS,
Lifecycle), анализатор использует эту таблицу как дополнительный
контекст. Каждый факт — проверенное утверждение с URL.

КАК РАСШИРЯТЬ:
- Добавь новый CuratedFact в FACTS ниже.
- keywords — список фраз/слов из требования ТЗ, при которых факт релевантен.
- platforms — список платформ, которым факт применим ("ГосОблако",
  "Облако VMware", "Advanced", "Evolution") или ["all"].
- statement — короткое утверждение (1-2 предложения), что Cloud.ru
  предоставляет.
- url — ссылка на cloud.ru/docs или cloud.ru/products.
- verified_on — дата сверки (YYYY-MM-DD).

КАК ИСПОЛЬЗОВАТЬ:
    from src.knowledge import find_relevant_facts, format_facts_for_prompt
    facts = find_relevant_facts(requirement_text, platform=None, limit=3)
    snippet = format_facts_for_prompt(facts)

ВАЖНО:
- Не используй curated_facts как замену RAG — это дополнение для случаев,
  когда RAG молчит. Если RAG возвращает релевантные чанки — используй их.
- Никогда не врать. Если фича есть «частично» — так и пиши в statement.
- Если фичи НЕТ в портфеле Cloud.ru — НЕ добавляй сюда (мы перечисляем
  только подтверждённые капабилити).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass
class CuratedFact:
    """Один проверенный факт о капабилити Cloud.ru."""

    keywords: list[str]              # Триггерные фразы из требования ТЗ
    platforms: list[str]             # Платформы, к которым применим
    statement: str                   # Короткое утверждение
    url: str                         # URL документации
    verified_on: str = ""            # Дата последней сверки (YYYY-MM-DD)
    title: str = ""                  # Человекочитаемое название источника
    related_concepts: list[str] = field(default_factory=list)  # Расширенный поиск
    # Programmatic enforcement: если LLM поставил NC, но curated_fact
    # явно подтверждает фичу — поднимаем до verdict из этого поля.
    # Значения: "match" / "partial" / "needs_clarification" / "" (не enforce).
    enforce_verdict: str = ""
    # Дополнительные ключевые слова, при которых enforce срабатывает —
    # узкая подсказка для случаев, когда keywords слишком общие.
    enforce_when_any: list[str] = field(default_factory=list)

    def applies_to_platform(self, platform: str) -> bool:
        """True, если факт применим к указанной платформе."""
        if not self.platforms:
            return False
        if "all" in self.platforms:
            return True
        return platform in self.platforms

    def matches_text(self, text: str) -> bool:
        """True, если текст требования совпадает с keywords или
        related_concepts факта."""
        if not text:
            return False
        norm = (text or "").lower().replace("ё", "е")
        for kw in self.keywords + self.related_concepts:
            if kw and kw.lower() in norm:
                return True
        return False

    def matches_enforce_context(self, text: str) -> bool:
        """Дополнительная проверка для enforcement: если enforce_when_any
        пуст — enforcement применяется при любом matches_text. Иначе —
        требуется один из enforce_when_any."""
        if not self.enforce_when_any:
            return True
        norm = (text or "").lower().replace("ё", "е")
        return any(t.lower() in norm for t in self.enforce_when_any)


# ----------------------------------------------------------------------
# КАТАЛОГ. Сверено по cloud.ru/docs в мае 2026.
# ----------------------------------------------------------------------

FACTS: list[CuratedFact] = [
    # --- Объектное хранилище OBS / S3 на Advanced ---
    CuratedFact(
        keywords=["worm", "write once read many", "иммутаб", "immutab"],
        platforms=["Advanced"],
        statement=(
            "Cloud.ru Advanced Object Storage Service (OBS) поддерживает "
            "WORM-режим: бакет можно перевести в режим WORM, объекты "
            "защищены от удаления и перезаписи на заданный период "
            "хранения. При включении WORM автоматически включается "
            "версионирование."
        ),
        url="https://cloud.ru/docs/obs/ug/topics/guides__worm",
        verified_on="2026-05-11",
        title="OBS — Защита WORM",
        related_concepts=["object lock", "retention", "compliance mode"],
        enforce_verdict="match",
        # Защита от ложных триггеров: «WORM» в тексте может быть просто
        # частью имени услуги («услуги S3 WORM»), а не запросом фичи.
        # Enforce только если рядом есть слова о СУТИ фичи: защита от
        # удаления, retention, write once, неизменяемость.
        enforce_when_any=[
            "защит", "удален", "удалит", "перезапис", "retention",
            "иммутаб", "immutab", "write once", "неизменя",
            "блокировк", "object lock", "гарантирован защит",
            "от изменений", "от удаления",
        ],
    ),
    # Governance/Legal Hold ставим ПЕРЕД Object Lock (Retention/Compliance) —
    # чтобы для требований с явным упоминанием Governance/Legal Hold
    # find_enforce_verdict сначала находил partial-факт, а не match.
    CuratedFact(
        keywords=["governance mode", "legal hold", "юридическая блокировка",
                  "бессрочная блокировка"],
        platforms=["Advanced"],
        statement=(
            "Governance Mode (привилегированные пользователи могут снимать "
            "блокировку или уменьшать срок) и Legal Hold (бессрочная "
            "юридическая блокировка) — в публичной документации Cloud.ru "
            "Advanced отдельно не описаны. Это **partial**: ставь partial "
            "с reasoning 'требуется уточнение у клиентского менеджера "
            "Cloud.ru'. НЕ ставь match только потому, что Object Lock в "
            "общем поддерживается."
        ),
        url="https://cloud.ru/docs/obs/ug/topics/guides__worm",
        verified_on="2026-05-11",
        title="OBS — Governance Mode / Legal Hold (partial)",
        related_concepts=["object lock", "worm"],
        enforce_verdict="partial",
    ),
    CuratedFact(
        keywords=["object lock", "retention period", "блокировка объект",
                  "срок хранения объект", "запрет на удаление",
                  "запрещает перезапись", "запрещает удаление",
                  "compliance mode"],
        platforms=["Advanced"],
        statement=(
            "Cloud.ru Advanced OBS поддерживает Object Lock: "
            "(а) Retention Period — настройка фиксированного срока (в днях "
            "или годах) запрета на удаление/перезапись = **match**; "
            "(б) Compliance Mode — запрет на удаление любым пользователем "
            "до истечения срока = **match**."
        ),
        url="https://cloud.ru/docs/obs/ug/topics/guides__worm",
        verified_on="2026-05-11",
        title="OBS — Object Lock (Retention/Compliance Mode)",
        related_concepts=["worm", "retention", "compliance mode"],
        enforce_verdict="match",
    ),
    CuratedFact(
        keywords=["versioning", "версионирование", "версии объект"],
        platforms=["Advanced", "Evolution"],
        statement=(
            "Cloud.ru OBS поддерживает версионирование бакетов "
            "(Versioning). Включается на уровне бакета. После включения "
            "версионирование можно приостановить (suspend), но не "
            "полностью отключить."
        ),
        url="https://cloud.ru/docs/obs/ug/topics/guides__bucket",
        verified_on="2026-05-11",
        title="OBS — Базовые настройки бакета",
        related_concepts=["object lock", "worm"],
        enforce_verdict="match",
    ),
    # SFTP-доступ к Cloud.ru S3 публично не подтверждён. Ставим раньше
    # PFS-факта, чтобы для требования «Поддержка SFTP» вердикт был NC,
    # а не partial (PFS keywords могут зацепить «файлов»).
    CuratedFact(
        keywords=["sftp", "поддержка sftp", "sftp-доступ", "sftp доступ",
                  "transfer family"],
        platforms=["all"],
        statement=(
            "SFTP-сервис для Object Storage (аналог AWS Transfer Family) "
            "у Cloud.ru в публичной документации НЕ подтверждён. Это "
            "**needs_clarification** — требуется уточнение у клиентского "
            "менеджера Cloud.ru, возможно ли решение через индивидуальный "
            "проект или партнёрский шлюз. НЕ ставь match только потому "
            "что S3 поддерживается."
        ),
        url="https://cloud.ru/docs/obs/ug/index",
        verified_on="2026-05-11",
        title="OBS — SFTP (не публично, NC)",
        related_concepts=["s3 transfer"],
        enforce_verdict="needs_clarification",
    ),
    CuratedFact(
        keywords=["pfs", "posix", "s3fs", "файловый протокол доступа",
                  "файловый доступ", "obsfs", "fuse", "mountpoint",
                  "файлов протокол"],
        platforms=["Advanced"],
        statement=(
            "Cloud.ru Advanced OBS поддерживает Parallel File System (PFS) "
            "и клиент obsfs — это FUSE-маунт, обеспечивающий POSIX-подобный "
            "доступ к бакету как к файловой системе. Это НЕ классический "
            "файловый протокол (NFS/SMB/SFTP) — это специфический клиент "
            "Cloud.ru. SFS Turbo — отдельный сервис NFS-доступа (через "
            "блочное хранилище, не S3). Поэтому для требования «файловый "
            "протокол доступа» — это **partial** с reasoning «есть PFS как "
            "FUSE-клиент, но это не классический NFS/SMB; уточнить у "
            "заказчика приемлем ли FUSE-вариант». НЕ ставь match только "
            "потому что PFS существует — это не то же самое."
        ),
        url="https://cloud.ru/docs/obs/ug/topics/guides__pfs",
        verified_on="2026-05-11",
        title="OBS — Parallel File System (PFS / obsfs) — partial",
        related_concepts=["nfs", "smb"],
        enforce_verdict="partial",
    ),
    CuratedFact(
        keywords=["lifecycle", "жизненн", "автоматический переход", "автоудален",
                  "сроком жизни объект", "управлен сроком жизни"],
        platforms=["Advanced", "Evolution"],
        statement=(
            "Cloud.ru OBS поддерживает Lifecycle Policies (политики "
            "жизненного цикла): автоматический переход объектов между "
            "классами хранения (Standard → Warm → Cold), автоматическое "
            "удаление по TTL."
        ),
        url="https://cloud.ru/docs/obs/ug/topics/guides__life-cycle",
        verified_on="2026-05-11",
        title="OBS — Жизненный цикл объектов",
        related_concepts=["storage class", "cold", "archive"],
        enforce_verdict="match",
    ),
    CuratedFact(
        keywords=["cold", "холодное хранилище", "archive", "архивное",
                  "долгосрочное хранение", "редк доступ"],
        platforms=["Advanced", "Evolution"],
        statement=(
            "Cloud.ru OBS предоставляет класс хранения Cold (холодное) "
            "для архивных данных с редким доступом. Стандартное "
            "восстановление — часы; быстрый режим (expedited) — 1-5 минут "
            "как опция. Точное SLA на восстановление — в проектном "
            "договоре."
        ),
        url="https://cloud.ru/docs/obs/ug/topics/concepts__storage-class",
        verified_on="2026-05-11",
        title="OBS — Классы хранения",
        related_concepts=["lifecycle"],
    ),
    CuratedFact(
        keywords=["multi-az", "зон доступности", "зоны доступности", "az",
                  "репликация между зон", "репликация в зон", "две зоны",
                  "несколько зон", "геораспределен"],
        platforms=["Advanced"],
        statement=(
            "Cloud.ru Advanced имеет 4 зоны доступности (AZ) в регионе "
            "ru-moscow-1. OBS поддерживает классы хранения Multi-AZ и "
            "Single-AZ — Multi-AZ обеспечивает репликацию данных между "
            "несколькими зонами доступности."
        ),
        url="https://cloud.ru/docs/advanced/overview/az-and-endpoints",
        verified_on="2026-05-11",
        title="Cloud.ru Advanced — Регионы и зоны доступности",
        related_concepts=["region", "dr"],
        enforce_verdict="match",
    ),
    CuratedFact(
        keywords=["tier iii", "tier 3", "uptime institute",
                  "сертификат цод", "уровень цод", "надежность цод",
                  "надёжность цод", "certification of facility"],
        platforms=["all"],
        statement=(
            "Все ЦОД Cloud.ru сертифицированы по стандарту Tier III "
            "(Uptime Institute Certification of Facility). Доступность "
            "ЦОД — не менее 99,982%. Это атрибут компании Cloud.ru, "
            "распространяется на все 4 платформы."
        ),
        url="https://cloud.ru/docs/vdc/ug/topics/faq/common-questions/common-questions__reliability-vdc",
        verified_on="2026-05-11",
        title="Cloud.ru — Надёжность ЦОД Tier III",
        # ВАЖНО: убрали 'доступности' и 'sla' из keywords/related —
        # это слишком общие слова, цеплялись за «параметры доступности
        # услуг» (KPI ТЗ), что давало ложный match Прил.4 Мосгор.
        # Enforce только если рядом есть слова про ЦОД/Tier/Uptime.
        related_concepts=["99.982", "uptime"],
        enforce_verdict="match",
        enforce_when_any=[
            "цод", "tier", "uptime", "надежн", "надёжн",
            "certification of facility",
        ],
    ),
    CuratedFact(
        keywords=["цод в россии", "цод в рф", "локализация в рф",
                  "территория российской федерации", "цод на территории",
                  "ru-moscow", "расположение цод"],
        platforms=["all"],
        statement=(
            "Все ЦОД Cloud.ru расположены на территории Российской "
            "Федерации. Это атрибут компании, применимо ко всем платформам."
        ),
        url="https://cloud.ru/products",
        verified_on="2026-05-11",
        title="Cloud.ru — Локализация в РФ",
        enforce_verdict="match",
    ),

    # --- SLA ---
    CuratedFact(
        keywords=["sla evolution object", "sla evolution", "sla ru-moscow"],
        platforms=["Evolution"],
        statement=(
            "Cloud.ru Evolution имеет публичную страницу SLA. Конкретный "
            "% SLA на Object Storage указан в приложении к договору; "
            "стандартная инфраструктурная доступность ЦОД 99,982%."
        ),
        url="https://cloud.ru/docs/evolution/overview/topics/sla",
        verified_on="2026-05-11",
        title="Cloud.ru Evolution — SLA",
    ),
    CuratedFact(
        keywords=["sla advanced", "sla на advanced"],
        platforms=["Advanced"],
        statement=(
            "Cloud.ru Advanced — публичная страница SLA. Точный SLA на "
            "OBS фиксируется в проектном договоре, инфраструктурная "
            "доступность ЦОД 99,982%."
        ),
        url="https://cloud.ru/docs/advanced/overview/sla",
        verified_on="2026-05-11",
        title="Cloud.ru Advanced — SLA",
    ),

    # --- Тарификация ---
    CuratedFact(
        keywords=["поминутн", "посекундн", "pay per second", "pay-per-second",
                  "пер минут", "пер секунд"],
        platforms=["all"],
        statement=(
            "Тарификация Cloud.ru — почасовая или помесячная. Поминутная "
            "и посекундная тарификации публично не подтверждены — "
            "если требование жёсткое, нужно уточнение у клиентского "
            "менеджера или специальные коммерческие условия."
        ),
        url="https://cloud.ru/documents/tariffs/evolution/object-storage",
        verified_on="2026-05-11",
        title="Cloud.ru — Тарифы Object Storage",
        related_concepts=["billing", "pricing"],
        enforce_verdict="needs_clarification",
    ),

    # --- Сеть / Direct Connect ---
    CuratedFact(
        keywords=["direct connect", "выделенный канал", "гарантированный канал",
                  "100 гбит", "10 гбит", "канал интернет"],
        platforms=["all"],
        statement=(
            "Cloud.ru Direct Connect — гарантированный канал интернета "
            "к платформам Cloud.ru до 100 Гбит/с. Применимо ко всем "
            "платформам через партнёрский / прямой канал связи."
        ),
        url="https://cloud.ru/products/direct-connect",
        verified_on="2026-05-11",
        title="Cloud.ru — Direct Connect",
    ),
    # --- Метрики качества сети (потери, задержка) — не декларируются публично ---
    CuratedFact(
        keywords=[
            "потер пакет", "процент потерянных пакет", "packet loss",
            "потеря пакет", "потеря трафика",
            "средняя сетевая задержк", "средняя задержк", "задержка не более",
            "задержк в сети", "latency",
            "jitter", "джиттер",
        ],
        platforms=["all"],
        statement=(
            "Конкретные метрики качества сети (потеря пакетов %, средняя "
            "сетевая задержка ms, jitter) Cloud.ru ПУБЛИЧНО НЕ ДЕКЛАРИРУЕТ. "
            "Для конкретного клиента эти параметры фиксируются в SLA "
            "Direct Connect или в проектном договоре. Это **needs_"
            "clarification** — требуется уточнение у сетевой службы Cloud.ru "
            "или запрос выписки из приложения SLA к договору. НЕ ставь "
            "match только потому, что Cloud.ru использует магистрали "
            "100 Гбит/с."
        ),
        url="https://cloud.ru/docs/dc/ug/topics/faq__bandwidth.html",
        verified_on="2026-05-11",
        title="Cloud.ru — Метрики качества сети (NC, не публично)",
        enforce_verdict="needs_clarification",
    ),

    # --- ИБ-сервисы ---
    CuratedFact(
        keywords=["шифрование at rest", "шифрование хранимых",
                  "шифрование данных", "encryption at rest", "sse"],
        platforms=["Advanced", "Evolution"],
        statement=(
            "Cloud.ru OBS поддерживает серверное шифрование (SSE) "
            "хранимых данных. Ключи управляются Cloud.ru. Клиентское "
            "шифрование с пользовательским ключом (SSE-C / BYOK) — "
            "требует уточнения."
        ),
        url="https://cloud.ru/docs/obs/ug/index",
        verified_on="2026-05-11",
        title="OBS — Шифрование",
    ),
    CuratedFact(
        keywords=["tls", "ssl", "https", "защищенное соединение",
                  "безопасное соединение", "шифрование в канале"],
        platforms=["all"],
        statement=(
            "Все API Cloud.ru доступны через HTTPS (TLS). Зашифрованное "
            "соединение — стандарт для всех платформ."
        ),
        url="https://cloud.ru/docs",
        verified_on="2026-05-11",
        title="Cloud.ru — TLS на API",
    ),

    # --- Аттестации / лицензии ---
    CuratedFact(
        keywords=["к1", "класс защищенности к1", "приказ фстэк 17",
                  "приказ фстэк №17"],
        platforms=["ГосОблако", "Облако VMware"],
        statement=(
            "ГосОблако (ГИС ГТ) и Облако VMware аттестованы по К1 "
            "(приказ ФСТЭК №17), что закрывает требования К1, К2, К3. "
            "Advanced и Evolution не имеют аттестации К1, только УЗ-1."
        ),
        url="https://cloud.ru/products/g-cloud",
        verified_on="2026-05-11",
        title="ГосОблако — К1, КИИ-1",
    ),
    CuratedFact(
        keywords=["уз-1", "уровень защищенности 1", "пдн", "152-фз"],
        platforms=["all"],
        statement=(
            "Все 4 платформы Cloud.ru аттестованы по УЗ-1 (ПП РФ №1119) "
            "и 152-ФЗ. УЗ-1 покрывает требования УЗ-2, УЗ-3, УЗ-4."
        ),
        url="https://cloud.ru/products",
        verified_on="2026-05-11",
        title="Cloud.ru — Аттестации по ПДн",
    ),
    CuratedFact(
        keywords=["фстэк лицензия", "тзки", "лицензия на тзки",
                  "л024-00107"],
        platforms=["all"],
        statement=(
            "Cloud.ru имеет лицензию ФСТЭК на ТЗКИ № Л024-00107-00/00582618 "
            "от 11.09.2019. Лицензия принадлежит компании, действует на "
            "все 4 платформы."
        ),
        url="https://cloud.ru/about",
        verified_on="2026-05-11",
        title="Cloud.ru — Лицензия ФСТЭК",
    ),
    CuratedFact(
        keywords=["роскомнадзор", "лицензия на связь", "лицензия услуги связи",
                  "171948", "171946", "171949", "оператор связи"],
        platforms=["all"],
        statement=(
            "Cloud.ru имеет лицензии Роскомнадзора на услуги связи "
            "№ 171948 / 171946 / 171949 от 07.02.2019. Лицензии "
            "принадлежат компании, действуют на все 4 платформы."
        ),
        url="https://cloud.ru/about",
        verified_on="2026-05-11",
        title="Cloud.ru — Лицензии Роскомнадзора",
    ),
    CuratedFact(
        keywords=["iso 27001", "исо 27001", "iec 27001", "гост 27001",
                  "27001:2022"],
        platforms=["all"],
        statement=(
            "Cloud.ru имеет сертификат ISO/IEC 27001:2022. Эквивалент "
            "ГОСТ Р ИСО/МЭК 27001 — требует юридического обоснования "
            "эквивалентности."
        ),
        url="https://cloud.ru/about",
        verified_on="2026-05-11",
        title="Cloud.ru — ISO 27001",
    ),

    # --- Поддержка / Service Desk ---
    CuratedFact(
        keywords=["service desk", "техническая поддержка", "24x7", "24/7",
                  "круглосуточн", "техподдержка", "support portal"],
        platforms=["all"],
        statement=(
            "Cloud.ru предоставляет круглосуточную техническую поддержку "
            "(24/7/365) через портал Service Desk и контактные каналы. "
            "Стандартные тарифы — Standard, Business, Premium."
        ),
        url="https://cloud.ru/docs/support",
        verified_on="2026-05-11",
        title="Cloud.ru — Техническая поддержка",
    ),

    # --- VMware-специфичные капабилити ---
    CuratedFact(
        keywords=["vcloud director", "vmware", "vsphere", "live migration",
                  "vmotion"],
        platforms=["Облако VMware"],
        statement=(
            "Облако VMware Cloud.ru построено на vCloud Director (на "
            "VMware vSphere), поддерживает Live Migration (vMotion), "
            "до 1 ТБ RAM на одну ВМ, SLA 99,982%."
        ),
        url="https://cloud.ru/products/vmware",
        verified_on="2026-05-11",
        title="Cloud.ru — Облако VMware",
    ),

    # --- Advanced / OpenStack-специфичные ---
    CuratedFact(
        keywords=["openstack", "evs", "ecs", "elastic compute",
                  "elastic volume"],
        platforms=["Advanced"],
        statement=(
            "Cloud.ru Advanced построен на стеке OpenStack. Сервисы — "
            "ECS (Elastic Cloud Server), EVS (Elastic Volume Service), "
            "OBS, ELB, VPC. Поддерживает высокую ёмкость ВМ (RAM > 1 ТБ)."
        ),
        url="https://cloud.ru/products/advanced",
        verified_on="2026-05-11",
        title="Cloud.ru Advanced — Архитектура",
    ),
]


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


_WORD_RE = re.compile(r"[A-Za-zА-ЯЁа-яё0-9-]+")


def _normalize(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


def _matches_keyword(requirement_text: str, keyword: str) -> bool:
    """True, если keyword содержится в тексте требования (case-insensitive)."""
    norm_req = _normalize(requirement_text)
    norm_kw = _normalize(keyword)
    if not norm_kw:
        return False
    return norm_kw in norm_req


def _score_fact(fact: CuratedFact, requirement_text: str, platform: str | None) -> int:
    """Скор релевантности: сколько ключевых слов сработало."""
    score = 0
    for kw in fact.keywords:
        if _matches_keyword(requirement_text, kw):
            score += 2  # точное keyword — приоритет
    for concept in fact.related_concepts:
        if _matches_keyword(requirement_text, concept):
            score += 1
    # Bonus, если требование явно про эту платформу
    if platform:
        if platform in fact.platforms or "all" in fact.platforms:
            score += 1
        elif fact.platforms and platform not in fact.platforms:
            # Платформа не совпадает — снижаем приоритет (но не до 0)
            score = max(0, score - 1)
    return score


def find_relevant_facts(
    requirement_text: str,
    platform: str | None = None,
    limit: int = 3,
) -> list[CuratedFact]:
    """Возвращает топ-N curated фактов, релевантных требованию.

    Args:
        requirement_text: текст требования из ТЗ.
        platform: имя платформы Cloud.ru или None (universal search).
        limit: сколько максимум фактов вернуть.
    """
    scored = []
    for fact in FACTS:
        score = _score_fact(fact, requirement_text, platform)
        if score > 0:
            scored.append((score, fact))
    scored.sort(key=lambda x: -x[0])
    return [f for _, f in scored[:limit]]


def find_enforce_verdict(
    requirement_text: str,
    platform: str,
) -> tuple[str, CuratedFact] | None:
    """Возвращает (enforce_verdict, fact), если curated_fact с непустым
    enforce_verdict применим к этому требованию и платформе.

    Иначе — None.

    Используется анализатором для programmatic enforcement: если LLM
    поставил «слабый» verdict (NC), а curated подтверждает фичу с
    enforce_verdict="match" — поднимаем до match с цитированием URL.

    Алгоритм:
      1. Перебираем FACTS в порядке списка.
      2. Берём первый, у которого:
         • applies_to_platform(platform) == True;
         • matches_text(requirement_text) == True;
         • enforce_verdict != "" (есть жёсткое правило);
         • matches_enforce_context(requirement_text) == True
           (если задан enforce_when_any — он должен совпасть).
      3. Возвращаем его enforce_verdict + сам факт.

    Это последовательный поиск, а не лучший — для определённости вердикта.
    Поэтому в FACTS более узкие правила (например, Governance/Legal Hold
    с partial) должны идти ВПЕРЁД более общих (Object Lock с match).
    """
    if not requirement_text or not platform:
        return None
    for fact in FACTS:
        if not fact.enforce_verdict:
            continue
        if not fact.applies_to_platform(platform):
            continue
        if not fact.matches_text(requirement_text):
            continue
        if not fact.matches_enforce_context(requirement_text):
            continue
        return fact.enforce_verdict, fact
    return None


def format_facts_for_prompt(facts: list[CuratedFact]) -> str:
    """Форматирует список фактов как текстовый блок для подмешивания в
    LLM-контекст. Используется как fallback, когда RAG не вернул чанков.
    """
    if not facts:
        return ""
    lines = ["[Дополнительный контекст — проверенные капабилити Cloud.ru]"]
    for i, fact in enumerate(facts, start=1):
        platforms_str = ", ".join(fact.platforms) if fact.platforms != ["all"] else "Все платформы"
        lines.append(
            f"({i}) [{platforms_str}] {fact.title or fact.url}\n"
            f"    {fact.statement}\n"
            f"    Источник: {fact.url}"
        )
    return "\n".join(lines)
