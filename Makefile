.PHONY: install install-intel-xpu run-slot train promote fmt


PY?=python
SLOT?=current
PORT?=8000


install:
$(PY) -m pip install -r requirements.txt


install-intel-xpu:
-$(PY) -m pip install -r requirements-intel-xpu.txt


run-slot:
bash scripts/run_slot.sh $(SLOT) $(PORT)


train:
bash scripts/train_from_scratch.sh


promote:
bash scripts/promote.sh $(SLOT) $(PORT)


fmt:
@echo "(建議在你環境安裝 black/ruff 等工具進行格式化)"
