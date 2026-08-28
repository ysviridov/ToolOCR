# Stage 2.2 — визуальный контроль preprocessing индекса

В ROI preview Test UI добавлена диагностическая полоса `OCR PREP AFTER stencil_dot_suppression_v1`.

Полоса строится по запросу и не сохраняется на диск. Для каждой из шести индексных ячеек отображается финальный бинарный canvas `96x128`, который после `stencil_dot_suppression_v1`, tight glyph crop и нормализации передаётся в single-digit Tesseract.

Для каждой ячейки показываются:

- `D1..D6` и распознанная цифра;
- confidence, если Tesseract вернул его;
- число удалённых компонент `removed`;
- число восстановленных слабых компонент около рукописного штриха `restored`;
- доля удалённого foreground `ink`.

Если preprocessing не оставил достаточного foreground, tile показывает `NO FOREGROUND`.

Визуализация не меняет алгоритм stencil detector, digit-cell geometry, orientation или итоговые OCR-данные. Она предназначена только для corpus-validation и настройки preprocessing.
