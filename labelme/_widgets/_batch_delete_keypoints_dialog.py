"""Batch Delete Keypoints Dialog.

A dialog for batch-deleting keypoints (point-type shapes) across multiple
annotation files, with spatial group selection and image preview.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple
from typing import cast

import numpy as np
from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from .._label_file import read_label_file
from .._shape import Shape
from .._shape import ShapeType

# ---------------------------------------------------------------------------
#  Data helpers
# ---------------------------------------------------------------------------


class _GroupInfo(NamedTuple):
    group_id: int
    center_x: float


class _FileEntry(NamedTuple):
    path: Path
    stem: str  # filename without extension


def _collect_files_in_dir(directory: str) -> list[_FileEntry]:
    """Return JSON annotation files in *directory*, sorted by stem."""
    dir_path = Path(directory)
    entries: list[_FileEntry] = []
    for p in sorted(dir_path.glob("*.json")):
        try:
            with open(p, encoding="utf-8") as fh:
                raw = json.load(fh)
            if "shapes" in raw and "imagePath" in raw:
                entries.append(_FileEntry(path=p, stem=p.stem))
        except Exception:
            pass
    return entries


def _find_image(json_path: Path, image_path: str) -> str | None:
    """Resolve the image file referenced by an annotation."""
    candidates = [
        json_path.parent / image_path,
        json_path.parent / (json_path.stem + ".jpg"),
        json_path.parent / (json_path.stem + ".png"),
        json_path.parent / (json_path.stem + ".jpeg"),
        json_path.parent / (json_path.stem + ".JPG"),
        json_path.parent / (json_path.stem + ".PNG"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _shapes_from_dicts(shape_dicts: list) -> list[Shape]:
    """Convert annotation shape dicts to Shape objects."""
    shapes: list[Shape] = []
    for sd in shape_dicts:
        shape = Shape(
            label=sd["label"],
            shape_type=cast(ShapeType, sd["shape_type"]),
            group_id=sd["group_id"],
            description=sd.get("description", ""),
            mask=sd.get("mask"),
            points=np.array(sd["points"], dtype=np.float64),
            closed=True,
        )
        shape.flags = sd.get("flags", {})
        shape.other_data = sd.get("other_data", {})
        shapes.append(shape)
    return shapes


def _read_shapes_fast(json_path: Path) -> list[dict]:
    """Read raw shape dicts from JSON without loading image data. Fast."""
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("shapes", [])


def _read_shapes(json_path: Path) -> list[Shape]:
    """Read shapes from a LabelMe JSON file, returning Shape objects."""
    annotation = read_label_file(str(json_path))
    return _shapes_from_dicts(annotation.shapes)


def _compute_groups(shapes: list[Shape]) -> list[_GroupInfo]:
    """Compute groups sorted by center X (left to right)."""
    groups: dict[int, list[float]] = {}
    for s in shapes:
        if s.group_id is None:
            continue
        gid = s.group_id
        if gid not in groups:
            groups[gid] = []
        if len(s.points) > 0:
            groups[gid].append(float(np.mean(s.points[:, 0])))
    result = [
        _GroupInfo(group_id=gid, center_x=float(np.mean(xs)))
        for gid, xs in groups.items()
    ]
    result.sort(key=lambda g: g.center_x)
    return result


def _collect_point_labels(shapes: list[Shape]) -> list[str]:
    """Return sorted unique labels from point-type shapes."""
    labels = sorted(
        {s.label for s in shapes if s.shape_type == "point" and s.label is not None}
    )
    return labels


# ---------------------------------------------------------------------------
#  Preview canvas (QGraphicsView-based)
# ---------------------------------------------------------------------------


class _PreviewScene(QtWidgets.QGraphicsScene):
    """Scene holding the pixmap and overlay items."""

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._pixmap_item: QtWidgets.QGraphicsPixmapItem | None = None
        self._overlay_items: list[QtWidgets.QGraphicsItem] = []

    def set_image(self, pixmap: QtGui.QPixmap) -> None:
        self.clear()
        self._pixmap_item = self.addPixmap(pixmap)
        self.setSceneRect(QtCore.QRectF(pixmap.rect()))

    def clear_overlays(self) -> None:
        for item in self._overlay_items:
            self.removeItem(item)
        self._overlay_items.clear()

    def add_overlay_item(self, item: QtWidgets.QGraphicsItem) -> None:
        self.addItem(item)
        self._overlay_items.append(item)


class _PreviewView(QtWidgets.QGraphicsView):
    """Zoomable / pannable image preview."""

    zoom_changed = QtCore.Signal(float)

    _ZOOM_FACTOR: float = 1.15
    _MIN_ZOOM: float = 0.05
    _MAX_ZOOM: float = 20.0

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = _PreviewScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._zoom_level: float = 1.0

    def set_preview_image(self, pixmap: QtGui.QPixmap) -> None:
        self._scene.set_image(pixmap)

    def clear_overlays(self) -> None:
        self._scene.clear_overlays()

    def add_overlay_item(self, item: QtWidgets.QGraphicsItem) -> None:
        self._scene.add_overlay_item(item)

    def zoom_level(self) -> float:
        return self._zoom_level

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y()
        factor = self._ZOOM_FACTOR if delta > 0 else 1.0 / self._ZOOM_FACTOR
        new_zoom = self._zoom_level * factor
        if self._MIN_ZOOM <= new_zoom <= self._MAX_ZOOM:
            self.scale(factor, factor)
            self._zoom_level = new_zoom
            self.zoom_changed.emit(self._zoom_level)
        event.accept()

    def fit_image(self) -> None:
        if self._scene._pixmap_item is None:
            return
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_level = self.transform().m11()
        self.zoom_changed.emit(self._zoom_level)


# ---------------------------------------------------------------------------
#  Main dialog
# ---------------------------------------------------------------------------

GROUP_COLORS: list[tuple[int, int, int]] = [
    (0, 200, 0),  # green
    (220, 50, 50),  # red
    (50, 50, 220),  # blue
    (220, 150, 0),  # orange
    (150, 0, 220),  # purple
    (0, 180, 180),  # teal
    (180, 180, 0),  # olive
    (220, 0, 150),  # magenta
]
HIGHLIGHT_COLOR: tuple[int, int, int] = (255, 60, 60)
GHOST_COLOR: tuple[int, int, int] = (80, 80, 80)


class BatchDeleteKeypointsDialog(QtWidgets.QDialog):
    """Batch delete keypoints across multiple annotation files."""

    def __init__(
        self,
        current_file: str | None = None,
        file_list: list[str] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("批量删除关键点")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 650)

        # ---- state ----
        self._current_file: str | None = current_file
        self._file_entries: list[_FileEntry] = []
        self._selected_stems: set[str] = set()
        self._current_entry: _FileEntry | None = None
        self._current_shapes: list[Shape] = []
        self._current_image_path: str | None = None  # cached for preview
        self._groups: list[_GroupInfo] = []
        self._target_group_id: int | None = None  # None = all groups
        self._all_point_labels: list[str] = []
        self._delete_labels: set[str] = set()

        # If a file list is given, use it; otherwise scan the current file's dir
        if file_list:
            self._file_entries = [
                _FileEntry(path=Path(f), stem=Path(f).stem) for f in file_list
            ]
        elif current_file:
            self._file_entries = _collect_files_in_dir(str(Path(current_file).parent))

        # Pre-select all files
        self._selected_stems = {e.stem for e in self._file_entries}

        # Build UI
        self._setup_ui()

        # Load initial state
        if self._file_entries:
            self._refresh_file_list()
            # Find current file index to start at
            start_idx = 0
            if current_file:
                cur_stem = Path(current_file).stem
                for i, e in enumerate(self._file_entries):
                    if e.stem == cur_stem:
                        start_idx = i
                        break
            self._navigate_to(start_idx)
            self._refresh_labels()
        else:
            self._frame_label.setText("未找到标注文件")

    # ------------------------------------------------------------------
    #  UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # --- left: preview ---
        left_panel = QtWidgets.QVBoxLayout()

        self._preview = _PreviewView(self)
        self._preview.zoom_changed.connect(self._on_zoom_changed)
        left_panel.addWidget(self._preview, 1)

        # zoom / nav bar
        nav_bar = QtWidgets.QHBoxLayout()

        self._zoom_label = QtWidgets.QLabel("100%")
        nav_bar.addWidget(self._zoom_label)

        nav_bar.addStretch()

        self._frame_label = QtWidgets.QLabel("0 / 0")
        nav_bar.addWidget(self._frame_label)

        left_panel.addLayout(nav_bar)

        main_layout.addLayout(left_panel, 3)

        # --- right: controls ---
        right_panel = QtWidgets.QVBoxLayout()
        right_panel.setSpacing(8)

        # File list
        file_group = QtWidgets.QGroupBox("文件列表")
        file_layout = QtWidgets.QVBoxLayout(file_group)

        file_btn_row = QtWidgets.QHBoxLayout()
        self._select_all_btn = QtWidgets.QPushButton("全选")
        self._select_all_btn.clicked.connect(self._select_all_files)
        file_btn_row.addWidget(self._select_all_btn)

        self._deselect_all_btn = QtWidgets.QPushButton("取消全选")
        self._deselect_all_btn.clicked.connect(self._deselect_all_files)
        file_btn_row.addWidget(self._deselect_all_btn)

        file_layout.addLayout(file_btn_row)

        self._file_list = QtWidgets.QListWidget()
        self._file_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._file_list.itemChanged.connect(self._on_file_check_changed)
        self._file_list.itemClicked.connect(self._on_file_clicked)
        file_layout.addWidget(self._file_list)

        right_panel.addWidget(file_group, 2)

        # Group selection
        self._group_group = QtWidgets.QGroupBox("目标组（按水平位置从左到右排序）")
        self._group_layout = QtWidgets.QVBoxLayout(self._group_group)
        self._group_buttons: list[QtWidgets.QRadioButton] = []
        self._group_button_group = QtWidgets.QButtonGroup(self)
        self._group_button_group.setExclusive(True)
        self._group_button_group.idToggled.connect(self._on_group_changed)
        right_panel.addWidget(self._group_group, 1)

        # Label selection
        self._label_group = QtWidgets.QGroupBox("要删除的关键点标签")
        label_outer = QtWidgets.QVBoxLayout(self._label_group)

        label_btn_row = QtWidgets.QHBoxLayout()
        self._select_all_labels_btn = QtWidgets.QPushButton("全选")
        self._select_all_labels_btn.clicked.connect(self._select_all_labels)
        label_btn_row.addWidget(self._select_all_labels_btn)
        self._deselect_all_labels_btn = QtWidgets.QPushButton("取消全选")
        self._deselect_all_labels_btn.clicked.connect(self._deselect_all_labels)
        label_btn_row.addWidget(self._deselect_all_labels_btn)
        label_outer.addLayout(label_btn_row)

        self._label_scroll = QtWidgets.QScrollArea()
        self._label_scroll.setWidgetResizable(True)
        self._label_check_widget = QtWidgets.QWidget()
        self._label_check_layout = QtWidgets.QVBoxLayout(self._label_check_widget)
        self._label_check_layout.setContentsMargins(0, 0, 0, 0)
        self._label_scroll.setWidget(self._label_check_widget)
        label_outer.addWidget(self._label_scroll, 1)

        self._label_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        right_panel.addWidget(self._label_group, 2)

        # Execute button
        btn_layout = QtWidgets.QHBoxLayout()

        self._preview_btn = QtWidgets.QPushButton("刷新预览")
        self._preview_btn.clicked.connect(self._refresh_preview)
        btn_layout.addWidget(self._preview_btn)

        self._execute_btn = QtWidgets.QPushButton("▶ 执行批量删除")
        self._execute_btn.setStyleSheet(
            "QPushButton { font-weight: bold; color: white; background-color: #c0392b; "
            "padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #e74c3c; }"
        )
        self._execute_btn.clicked.connect(self._execute_batch)
        btn_layout.addWidget(self._execute_btn)

        close_btn = QtWidgets.QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)

        right_panel.addLayout(btn_layout)

        main_layout.addLayout(right_panel, 2)

    # ------------------------------------------------------------------
    #  Keyboard shortcuts
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_D and not event.modifiers():
            self._navigate_relative(1)
        elif event.key() == Qt.Key.Key_A and not event.modifiers():
            self._navigate_relative(-1)
        elif (
            event.key() == Qt.Key.Key_F
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._preview.fit_image()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    #  File list
    # ------------------------------------------------------------------

    def _refresh_file_list(self) -> None:
        """Rebuild file list with check states."""
        self._file_list.blockSignals(True)
        self._file_list.clear()
        for entry in self._file_entries:
            item = QtWidgets.QListWidgetItem(entry.stem)
            item.setData(Qt.ItemDataRole.UserRole, entry)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if entry.stem in self._selected_stems
                else Qt.CheckState.Unchecked
            )
            self._file_list.addItem(item)
        self._file_list.blockSignals(False)

    def _select_all_files(self) -> None:
        self._selected_stems = {e.stem for e in self._file_entries}
        self._refresh_file_list()
        self._refresh_labels()

    def _deselect_all_files(self) -> None:
        self._selected_stems.clear()
        self._refresh_file_list()
        self._refresh_labels()

    def _on_file_check_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        """Sync checkbox state without navigating."""
        entry = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(entry, _FileEntry):
            return
        if item.checkState() == Qt.CheckState.Checked:
            self._selected_stems.add(entry.stem)
        else:
            self._selected_stems.discard(entry.stem)
        self._refresh_labels()

    def _on_file_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        """Navigate to clicked file."""
        row = self._file_list.row(item)
        if row >= 0:
            self._navigate_to(row)

    # ------------------------------------------------------------------
    #  Navigation
    # ------------------------------------------------------------------

    def _navigate_to(self, index: int) -> None:
        if index < 0 or index >= len(self._file_entries):
            return
        self._current_entry = self._file_entries[index]
        self._file_list.blockSignals(True)
        self._file_list.setCurrentRow(index)
        self._file_list.blockSignals(False)

        self._load_current_file()
        self._refresh_groups()
        self._refresh_preview()
        self._update_frame_label()

    def _navigate_relative(self, delta: int) -> None:
        if self._current_entry is None:
            return
        for i, e in enumerate(self._file_entries):
            if e.stem == self._current_entry.stem:
                new_idx = max(0, min(len(self._file_entries) - 1, i + delta))
                self._navigate_to(new_idx)
                return

    def _update_frame_label(self) -> None:
        if self._current_entry is None:
            self._frame_label.setText("0 / 0")
            return
        for i, e in enumerate(self._file_entries):
            if e.stem == self._current_entry.stem:
                self._frame_label.setText(f"{i + 1} / {len(self._file_entries)}")
                return

    # ------------------------------------------------------------------
    #  Data loading
    # ------------------------------------------------------------------

    def _load_current_file(self) -> None:
        if self._current_entry is None:
            return
        try:
            annotation = read_label_file(str(self._current_entry.path))
            self._current_shapes = _shapes_from_dicts(annotation.shapes)
            self._current_image_path = _find_image(
                self._current_entry.path, annotation.image_path
            )
        except Exception:
            self._current_shapes = []
            self._current_image_path = None

    def _refresh_groups(self) -> None:
        """Rebuild group radio buttons, preserving previous selection if possible."""
        prev_gid = self._target_group_id
        prev_rank: int | None = None
        if prev_gid is not None:
            for i, gi in enumerate(self._groups):
                if gi.group_id == prev_gid:
                    prev_rank = i
                    break

        # Clear old buttons
        for btn in self._group_buttons:
            self._group_button_group.removeButton(btn)
            self._group_layout.removeWidget(btn)
            btn.deleteLater()
        self._group_buttons.clear()

        self._groups = _compute_groups(self._current_shapes)
        n = len(self._groups)

        # "All groups" option
        all_btn = QtWidgets.QRadioButton(f"所有组（共 {n} 组）")
        self._group_button_group.addButton(all_btn, -1)
        self._group_layout.addWidget(all_btn)
        self._group_buttons.append(all_btn)

        target_id: int = -1  # default: all groups
        for rank, gi in enumerate(self._groups, 1):
            label = f"左侧第 {rank}/{n} 组  (group_id={gi.group_id})"
            btn = QtWidgets.QRadioButton(label)
            self._group_button_group.addButton(btn, gi.group_id)
            self._group_layout.addWidget(btn)
            self._group_buttons.append(btn)
            # Preserve: same group_id match
            if prev_gid is not None and gi.group_id == prev_gid:
                target_id = gi.group_id
            # Preserve: same rank match (if gid doesn't exist in new file)
            if target_id == -1 and prev_rank is not None and rank - 1 == prev_rank:
                target_id = gi.group_id

        if target_id == -1:
            all_btn.setChecked(True)
            self._target_group_id = None
        else:
            btn_to_check = self._group_button_group.button(target_id)
            if btn_to_check is not None:
                btn_to_check.setChecked(True)
                self._target_group_id = target_id
            else:
                all_btn.setChecked(True)
                self._target_group_id = None

    def _refresh_labels(self) -> None:
        """Collect point labels from all selected files and rebuild checkboxes."""
        # Clear old
        for cb in self._label_checkboxes.values():
            self._label_check_layout.removeWidget(cb)
            cb.deleteLater()
        self._label_checkboxes.clear()

        # Collect from all selected files (fast read, no image data)
        all_labels: set[str] = set()
        for entry in self._file_entries:
            if entry.stem not in self._selected_stems:
                continue
            try:
                raw_shapes = _read_shapes_fast(entry.path)
                for s in raw_shapes:
                    if s.get("shape_type") == "point" and s.get("label"):
                        all_labels.add(s["label"])
            except Exception:
                pass

        self._all_point_labels = sorted(all_labels)
        for lab in self._all_point_labels:
            cb = QtWidgets.QCheckBox(lab)
            cb.setChecked(lab in self._delete_labels)
            cb.toggled.connect(
                lambda checked, label=lab: self._on_label_toggled(label, checked)
            )
            self._label_check_layout.addWidget(cb)
            self._label_checkboxes[lab] = cb

        self._label_check_layout.addStretch()

    def _on_label_toggled(self, label: str, checked: bool) -> None:
        if checked:
            self._delete_labels.add(label)
        else:
            self._delete_labels.discard(label)
        self._refresh_preview()

    def _select_all_labels(self) -> None:
        self._label_check_widget.blockSignals(True)
        for cb in self._label_checkboxes.values():
            cb.setChecked(True)
        self._label_check_widget.blockSignals(False)
        self._delete_labels = set(self._all_point_labels)
        self._refresh_preview()

    def _deselect_all_labels(self) -> None:
        self._label_check_widget.blockSignals(True)
        for cb in self._label_checkboxes.values():
            cb.setChecked(False)
        self._label_check_widget.blockSignals(False)
        self._delete_labels.clear()
        self._refresh_preview()

    def _on_group_changed(self, gid: int, checked: bool = True) -> None:
        if not checked:
            return
        self._target_group_id = gid if gid >= 0 else None
        self._refresh_preview()

    def _on_zoom_changed(self, zoom: float) -> None:
        self._zoom_label.setText(f"{int(zoom * 100)}%")

    # ------------------------------------------------------------------
    #  Preview
    # ------------------------------------------------------------------

    def _refresh_preview(self) -> None:
        if self._current_entry is None:
            return

        shapes = self._current_shapes

        # Load image using cached path
        if self._current_image_path and Path(self._current_image_path).exists():
            pixmap = QtGui.QPixmap(self._current_image_path)
        else:
            pixmap = QtGui.QPixmap(1920, 1080)
            pixmap.fill(QtGui.QColor(40, 40, 40))

        self._preview.set_preview_image(pixmap)
        self._preview.clear_overlays()

        groups = _compute_groups(shapes)
        target_gid = self._target_group_id

        # Draw each shape
        for shape in shapes:
            if shape.shape_type == "point" and len(shape.points) == 0:
                continue
            if shape.label is None:
                continue

            gid = shape.group_id
            is_target = target_gid is None or (gid is not None and gid == target_gid)

            # Color
            if is_target:
                if shape.label in self._delete_labels:
                    color = HIGHLIGHT_COLOR
                else:
                    # Find rank for this group
                    rank = 0
                    for i, gi in enumerate(groups):
                        if gi.group_id == gid:
                            rank = i
                            break
                    color = GROUP_COLORS[rank % len(GROUP_COLORS)]
            else:
                color = GHOST_COLOR

            self._draw_shape_overlay(shape, color, is_target)

        self._preview.fit_image()

    def _draw_shape_overlay(
        self, shape: Shape, color: tuple[int, int, int], is_target: bool
    ) -> None:
        r, g, b = color
        qcolor = QtGui.QColor(r, g, b)
        pen = QtGui.QPen(qcolor)
        pen_width = 3 if is_target else 1
        pen.setWidth(pen_width)

        pts = shape.points
        n_pts = len(pts)

        if shape.shape_type == "rectangle" and n_pts >= 2:
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            rect = QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)
            item = self._preview._scene.addRect(rect, pen)
            self._preview.add_overlay_item(item)

        elif shape.shape_type == "point" and n_pts >= 1:
            x, y = pts[0]
            r_size = 5 if is_target else 3
            ellipse = QtCore.QRectF(x - r_size, y - r_size, r_size * 2, r_size * 2)
            brush = QtGui.QBrush(qcolor)
            item = self._preview._scene.addEllipse(ellipse, pen, brush)
            self._preview.add_overlay_item(item)

            if shape.label:
                text_item = self._preview._scene.addSimpleText(
                    shape.label, QtGui.QFont("sans-serif", 8)
                )
                text_item.setPos(x + 4, y - 12)
                text_item.setBrush(QtGui.QBrush(qcolor))
                self._preview.add_overlay_item(text_item)

        elif shape.shape_type == "polygon" and n_pts >= 2:
            polygon = QtGui.QPolygonF([QtCore.QPointF(p[0], p[1]) for p in pts])
            brush = QtGui.QBrush(QtGui.QColor(r, g, b, 40))
            item = self._preview._scene.addPolygon(polygon, pen, brush)
            self._preview.add_overlay_item(item)

        elif shape.shape_type == "circle" and n_pts >= 2:
            cx, cy = pts[0]
            px, py = pts[1]
            d = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            rect = QtCore.QRectF(cx - d, cy - d, d * 2, d * 2)
            item = self._preview._scene.addEllipse(rect, pen)
            self._preview.add_overlay_item(item)

        elif shape.shape_type == "line" and n_pts >= 2:
            line = QtCore.QLineF(
                QtCore.QPointF(pts[0][0], pts[0][1]),
                QtCore.QPointF(pts[1][0], pts[1][1]),
            )
            item = self._preview._scene.addLine(line, pen)
            self._preview.add_overlay_item(item)

        elif shape.shape_type == "linestrip" and n_pts >= 2:
            path = QtGui.QPainterPath()
            path.moveTo(pts[0][0], pts[0][1])
            for p in pts[1:]:
                path.lineTo(p[0], p[1])
            item = self._preview._scene.addPath(path, pen)
            self._preview.add_overlay_item(item)

        elif shape.shape_type == "oriented_rectangle" and n_pts >= 4:
            polygon = QtGui.QPolygonF(
                [
                    QtCore.QPointF(pts[0][0], pts[0][1]),
                    QtCore.QPointF(pts[1][0], pts[1][1]),
                    QtCore.QPointF(pts[2][0], pts[2][1]),
                    QtCore.QPointF(pts[3][0], pts[3][1]),
                ]
            )
            item = self._preview._scene.addPolygon(polygon, pen)
            self._preview.add_overlay_item(item)

        elif shape.shape_type == "mask" and n_pts >= 2:
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            rect = QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)
            dash_pen = QtGui.QPen(qcolor)
            dash_pen.setWidth(pen_width)
            dash_pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            item = self._preview._scene.addRect(rect, dash_pen)
            self._preview.add_overlay_item(item)

    # ------------------------------------------------------------------
    #  Batch execution
    # ------------------------------------------------------------------

    def _execute_batch(self) -> None:
        selected = [e for e in self._file_entries if e.stem in self._selected_stems]
        if not selected:
            QtWidgets.QMessageBox.warning(self, "提示", "没有选中的文件")
            return

        if not self._delete_labels:
            QtWidgets.QMessageBox.warning(
                self, "提示", "请至少选择一个要删除的关键点标签"
            )
            return

        target_gid = self._target_group_id  # None = all groups
        n_sel = len(selected)
        n_labels = len(self._delete_labels)
        labels_str = "、".join(sorted(self._delete_labels))
        gid_str = f"group_id={target_gid}" if target_gid is not None else "所有组"

        msg = (
            f"将对 {n_sel} 个文件执行批量删除:\n\n"
            f"  目标组: {gid_str}\n"
            f"  删除标签 ({n_labels} 个): {labels_str}\n\n"
            f"原始文件将被直接修改。确定执行？"
        )
        reply = QtWidgets.QMessageBox.question(
            self,
            "确认批量删除",
            msg,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        total_deleted = 0
        skipped = 0
        errors: list[str] = []

        progress = QtWidgets.QProgressDialog(
            "正在批量删除关键点…", "取消", 0, len(selected), self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(500)

        for i, entry in enumerate(selected):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            progress.setLabelText(f"处理中: {entry.stem}  ({i + 1}/{len(selected)})")
            QtWidgets.QApplication.processEvents()

            try:
                with open(entry.path, encoding="utf-8") as fh:
                    data = json.load(fh)

                shapes_raw = data.get("shapes", [])
                new_shapes = []
                for s in shapes_raw:
                    s_gid = s.get("group_id")
                    s_label = s.get("label", "")
                    s_type = s.get("shape_type", "")

                    # Check if this shape should be deleted
                    should_delete = (
                        s_type == "point"
                        and s_label in self._delete_labels
                        and (
                            target_gid is None
                            or (s_gid is not None and s_gid == target_gid)
                        )
                    )
                    if should_delete:
                        total_deleted += 1
                    else:
                        new_shapes.append(s)

                if len(new_shapes) != len(shapes_raw):
                    data["shapes"] = new_shapes
                    with open(entry.path, "w", encoding="utf-8") as fh:
                        json.dump(data, fh, ensure_ascii=False, indent=2)
            except Exception as exc:
                errors.append(f"{entry.stem}: {exc}")
                skipped += 1

        result_msg = (
            f"批量删除完成！\n"
            f"处理 {n_sel} 个文件，"
            f"成功 {n_sel - skipped} 个，"
            f"跳过 {skipped} 个，"
            f"删除 {total_deleted} 个关键点。"
        )
        if errors:
            result_msg += f"\n\n错误 ({len(errors)}):\n" + "\n".join(errors[:5])
            if len(errors) > 5:
                result_msg += f"\n... 及其他 {len(errors) - 5} 个错误"

        QtWidgets.QMessageBox.information(self, "完成", result_msg)

        progress.setValue(len(selected))

        # Refresh preview after delete
        self._load_current_file()
        self._refresh_groups()
        self._refresh_preview()
