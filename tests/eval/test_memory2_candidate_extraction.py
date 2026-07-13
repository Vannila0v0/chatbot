from eval.memory2_cluster.extract_candidates import sanitize_content


def test_sanitize_content_redacts_secret_before_label_and_long_account() -> None:
    content = (
        "4C6AFE737E2AD2271530740414B30C1F这是我的api key，"
        "这是我的steamid 76561198852785613"
    )

    sanitized = sanitize_content(content)

    assert "4C6AFE737E2AD2271530740414B30C1F" not in sanitized
    assert "76561198852785613" not in sanitized
    assert "[长十六进制凭据已脱敏]" in sanitized
    assert "[长数字账号已脱敏]" in sanitized


def test_sanitize_content_redacts_steam_user_and_friend_names() -> None:
    content = (
        "你 Steam 名叫 **ExampleUser**。\n"
        "正在游戏中：\n- **FriendOne** — Game A\n- **1234567890** — Game B"
    )

    sanitized = sanitize_content(content)

    assert "ExampleUser" not in sanitized
    assert "FriendOne" not in sanitized
    assert "1234567890" not in sanitized
    assert "[用户名已脱敏]" in sanitized
    assert "[好友昵称已脱敏]" in sanitized


def test_sanitize_content_keeps_game_names_in_library_summary() -> None:
    content = "Steam 游戏库：\n- **Apex Legends** — 100h\n还可以查看好友列表"

    sanitized = sanitize_content(content)

    assert "Apex Legends" in sanitized
