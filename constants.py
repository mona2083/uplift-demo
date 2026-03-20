"""Segment visuals, feature labels, and shared constants for the uplift demo."""

PORTFOLIO_URL = "https://mona2083.github.io/portfolio-2026/index.html"

SEGMENT_COLORS = {
    "Sure Things": "#2d6a4f",
    "Persuadables": "#1a4a7a",
    "Lost Causes": "#888888",
    "Sleeping Dogs": "#c0392b",
}
SEGMENT_LABELS_JA = {
    "Sure Things": "鉄板層",
    "Persuadables": "説得可能層",
    "Lost Causes": "無関心層",
    "Sleeping Dogs": "あまのじゃく層",
}
SEGMENT_DESC = {
    "en": {
        "Sure Things": "Buy anyway — wasting coupons here",
        "Persuadables": "Buy only with coupon — target here",
        "Lost Causes": "Won't buy regardless",
        "Sleeping Dogs": "Coupon makes them leave",
    },
    "ja": {
        "Sure Things": "クーポンなしでも買う → 配ると利益の無駄",
        "Persuadables": "クーポンがあれば買う → ここだけに予算投下",
        "Lost Causes": "何をしても買わない",
        "Sleeping Dogs": "クーポンを送ると鬱陶しがって離脱",
    },
}
SEG_ORDER = ["Sure Things", "Persuadables", "Lost Causes", "Sleeping Dogs"]

FEAT_LABELS = {
    "age": "年齢 / Age",
    "past_spend": "過去購買額 / Past Spend",
    "visit_freq": "来店頻度 / Visit Freq",
    "days_since_last": "最終来店からの日数 / Recency",
    "avg_basket": "平均購買額 / Avg Basket",
}

PERSONAS = {
    "Sure Things": {
        "en": {
            "icon": "💎",
            "tagline": "Your loyal regulars",
            "traits": [
                "High visit frequency",
                "High past spend",
                "Short days since last visit",
                "Large average basket",
            ],
        },
        "ja": {
            "icon": "💎",
            "tagline": "あなたの常連客",
            "traits": [
                "来店頻度：高",
                "過去購買額：高",
                "最終来店日：直近",
                "平均購買額：大",
            ],
        },
    },
    "Persuadables": {
        "en": {
            "icon": "🎯",
            "tagline": "Price-sensitive occasional shoppers",
            "traits": [
                "Moderate visit frequency",
                "Medium past spend",
                "Longer days since last visit",
                "Responds to price incentives",
            ],
        },
        "ja": {
            "icon": "🎯",
            "tagline": "価格に敏感な不定期客",
            "traits": [
                "来店頻度：中程度",
                "過去購買額：中程度",
                "最終来店日：やや遠い",
                "価格インセンティブに反応",
            ],
        },
    },
    "Lost Causes": {
        "en": {
            "icon": "🚪",
            "tagline": "Window shoppers, not buyers",
            "traits": [
                "Low visit frequency",
                "Low past spend",
                "Very long since last visit",
                "Not price-motivated",
            ],
        },
        "ja": {
            "icon": "🚪",
            "tagline": "見るだけ客",
            "traits": [
                "来店頻度：低",
                "過去購買額：低",
                "最終来店日：遠い",
                "価格で動かない",
            ],
        },
    },
    "Sleeping Dogs": {
        "en": {
            "icon": "⚠️",
            "tagline": "VIPs who hate being sold to",
            "traits": [
                "Very high visit frequency",
                "Very high past spend",
                "Recent visits",
                "Resent promotional noise",
            ],
        },
        "ja": {
            "icon": "⚠️",
            "tagline": "売り込みを嫌うVIP客",
            "traits": [
                "来店頻度：非常に高",
                "過去購買額：非常に高",
                "最終来店日：直近",
                "販促メッセージを鬱陶しがる",
            ],
        },
    },
}
