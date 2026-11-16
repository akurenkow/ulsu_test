#!/usr/bin/env python
"""Импорт демо-дашборда с подстановкой пароля БД DWH.

Штатный ``superset import-dashboards`` не умеет передавать пароль: в экспорте
``databases/*.yaml`` пароль замаскирован (``XXXXXXXXXX``), поэтому на чистой
машине импорт падает с «Must provide a password for the database». Этот скрипт
извлекает реальный пароль из ``SUPERSET_DWH_URI`` и передаёт его импортёрам через
словарь ``passwords`` (ключ — путь к YAML БД).

Датасеты и чарты импортируются отдельно с ``overwrite=True``: встроенный
``ImportDashboardsCommand`` при повторном импорте пропускает уже существующие
датасеты/чарты (внутри у них ``overwrite=False``), поэтому без этой надстройки
правки колонок и метрик из ``demo.zip`` не применялись бы на поднятом инстансе.

Запускается из ``superset-init.sh``; вне контейнера не используется.
"""
from __future__ import annotations

import os
import zipfile
from urllib.parse import urlsplit

from flask import g

from superset import security_manager
from superset.app import create_app

BUNDLE = os.environ.get("SUPERSET_DASHBOARD_BUNDLE", "/opt/superset/dashboards/demo.zip")
DWH_URI = os.environ.get("SUPERSET_DWH_URI", "")
ADMIN_USER = os.environ.get("SUPERSET_ADMIN_USER", "admin")

METADATA = "metadata.yaml"


def _database_passwords(contents: dict[str, str]) -> dict[str, str]:
    """Пароль для каждого ``databases/*.yaml`` из URI окружения."""
    password = urlsplit(DWH_URI).password or ""
    if not password:
        return {}
    return {
        name: password
        for name in contents
        if name.startswith("databases/") and name.endswith(".yaml")
    }


def _with_metadata_type(contents: dict[str, str], type_: str) -> dict[str, str]:
    """Копия бандла с ``metadata.yaml`` нужного типа.

    У ``ImportDashboardsCommand`` тип метаданных — ``Dashboard``; для отдельных
    импортёров датасетов/чартов требуется тип ``SqlaTable`` / ``Slice``.
    """
    contents = dict(contents)
    meta = contents.get(METADATA, "version: 1.0.0\ntype: Dashboard\n")
    lines = []
    for line in meta.splitlines():
        lines.append(f"type: {type_}" if line.startswith("type:") else line)
    contents[METADATA] = "\n".join(lines) + "\n"
    return contents


def main() -> None:
    app = create_app()
    with app.app_context():
        # Импорт проверяет права (can_write на Database/Dataset/Dashboard),
        # поэтому устанавливаем пользователя-администратора.
        g.user = security_manager.find_user(username=ADMIN_USER)

        from superset.commands.chart.importers.dispatcher import ImportChartsCommand
        from superset.commands.dashboard.importers.dispatcher import (
            ImportDashboardsCommand,
        )
        from superset.commands.dataset.importers.dispatcher import (
            ImportDatasetsCommand,
        )
        from superset.commands.importers.v1.utils import get_contents_from_bundle

        with zipfile.ZipFile(BUNDLE) as bundle:
            contents = get_contents_from_bundle(bundle)

        passwords = _database_passwords(contents)

        # Штатный ``ImportDashboardsCommand`` обновляет только дашборд, а датасеты
        # и чарты при повторном импорте пропускает (внутри они импортируются с
        # ``overwrite=False``). Импортируем их отдельно с ``overwrite=True``, чтобы
        # изменения колонок/метрик из демо-экспорта применялись и на уже
        # поднятом инстансе, а не только на чистой машине.
        ImportDatasetsCommand(
            _with_metadata_type(contents, "SqlaTable"),
            overwrite=True,
            passwords=passwords,
        ).run()

        ImportChartsCommand(
            _with_metadata_type(contents, "Slice"),
            overwrite=True,
            passwords=passwords,
        ).run()

        ImportDashboardsCommand(
            _with_metadata_type(contents, "Dashboard"),
            overwrite=True,
            passwords=passwords,
        ).run()

        print(f"Imported dashboards from {BUNDLE}")


if __name__ == "__main__":
    main()
