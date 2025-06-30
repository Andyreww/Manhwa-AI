# app.py
# The main GUI application for the Manhwa Translator
# --- VERSION 28.0: Final Features ---

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, font 
import os
import threading
from PIL import Image, ImageTk
import sys
import json 
import re 
import time

# --- IMPORTANT: Import our processing logic ---
try:
    from processing import get_model, process_single_image
except ImportError:
    print("FATAL ERROR: processing.py not found. Make sure it's in the same folder as app.py")
    get_model, process_single_image = None, None


# --- Draggable File Item Frame ---
class DraggableFileItem(ctk.CTkFrame):
    def __init__(self, master, filename, reorder_callback, select_callback, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.filename = filename
        self.reorder_callback = reorder_callback
        self.select_callback = select_callback
        self.configure(fg_color="transparent")
        self.handle = ctk.CTkLabel(self, text="☰", cursor="hand2", text_color="gray")
        self.handle.pack(side="left", padx=(5, 10))
        self.filename_label = ctk.CTkLabel(self, text=filename, anchor="w", cursor="hand2")
        self.filename_label.pack(side="left", fill="x", expand=True)
        self.bind_all_children("<ButtonPress-1>", self.on_press)
        self.bind_all_children("<B1-Motion>", self.on_drag)
        self.bind_all_children("<ButtonRelease-1>", self.on_release)
        self._drag_start_y, self._placeholder, self._was_dragged = 0, None, False

    def bind_all_children(self, sequence, callback):
        self.bind(sequence, callback)
        for child in self.winfo_children():
            child.bind(sequence, callback)

    def on_press(self, event):
        self._drag_start_y, self._was_dragged = event.y, False
    
    def on_drag(self, event):
        if not self._was_dragged and abs(event.y - self._drag_start_y) > 3: self._was_dragged = True
        if not self._was_dragged: return

        if not self._placeholder:
            self.lift()
            self._placeholder = ctk.CTkFrame(self.master, height=2, fg_color="cyan")
        self.place(x=0, y=self.winfo_y() + event.y - self._drag_start_y, relwidth=1.0)
        y = self.winfo_y()
        items = [w for w in self.master.winfo_children() if isinstance(w, DraggableFileItem) and w is not self]
        new_index = sum(1 for item in items if y > item.winfo_y() + item.winfo_height() / 2)
        self.master._dnd_placeholder_index = new_index
        if self._placeholder: self._placeholder.pack_forget()
        for widget in items: widget.pack_forget()
        for i, widget in enumerate(items):
            if i == new_index: self._placeholder.pack(fill="x", padx=10, pady=1)
            widget.pack(fill="x", padx=10, pady=2)
        if new_index == len(items): self._placeholder.pack(fill="x", padx=10, pady=1)

    def on_release(self, event):
        if self._placeholder: self._placeholder.destroy(); self._placeholder = None
        if self.winfo_manager() == 'place': self.place_forget()
        if self._was_dragged: self.reorder_callback(self.filename, getattr(self.master, '_dnd_placeholder_index', 0))
        else:
            self.select_callback(self.filename)
            self.reorder_callback(None, None)

    def configure_appearance(self, is_selected):
        font_weight = "bold" if is_selected else "normal"
        accent_color_data = ctk.ThemeManager.theme["CTkButton"]["fg_color"]
        if isinstance(accent_color_data, (list, tuple)):
            accent_color = accent_color_data[0] if ctk.get_appearance_mode() == "Dark" else accent_color_data[1]
        else:
            accent_color = accent_color_data
        
        default_text_color = ctk.ThemeManager.theme["CTkLabel"]["text_color"]
        text_color = accent_color if is_selected else default_text_color
        
        current_font = self.filename_label.cget("font")
        self.filename_label.configure(font=ctk.CTkFont(family=current_font.cget("family"), size=current_font.cget("size"), weight=font_weight), text_color=text_color)

# --- Before/After Image Viewer ---
class BeforeAfterSlider(ctk.CTkFrame):
    def __init__(self, master, live_preview_var, performance_mode_var):
        super().__init__(master)
        self.app = master 
        self.live_preview_var = live_preview_var
        self.performance_mode_var = performance_mode_var

        self.grid_columnconfigure(0, weight=1); self.grid_rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.h_scroll = ctk.CTkScrollbar(self, orientation="horizontal", command=self.canvas.xview)
        self.original_image, self.translated_image, self.disp_orig, self.disp_trans = None, None, None, None
        self.photo_image_ref, self.image_id, self.divider_id = None, None, None
        self.zoom_ratio, self.fit_ratio = 1.0, 1.0
        
        self._after_id = None
        # --- Variables for scroll navigation ---
        self._is_navigating = False 
        self._boundary_hit = None 
        self._boundary_hit_time = 0

        self.bind("<Configure>", self.on_resize)
        self.slider = ctk.CTkSlider(self, from_=0, to=1, command=self.on_slider_move)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)
        
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        self.update_canvas_bg()

    def update_canvas_bg(self):
        try:
            bg_color = self._apply_appearance_mode(self._fg_color)
            self.canvas.configure(bg=bg_color)
        except Exception as e:
            self.canvas.configure(bg="#2B2B2B") 

    def _on_mousewheel(self, event):
        scroll_dir = 0
        if sys.platform == "darwin": scroll_dir = -1 * event.delta
        elif event.num == 4: scroll_dir = -1
        elif event.num == 5: scroll_dir = 1
        else: scroll_dir = -1 * (event.delta // 120)

        is_scrollable = self.disp_orig and self.disp_orig.height > self.winfo_height()

        if not is_scrollable:
            if len(self.app.image_files) > 1: self.check_for_image_navigation(scroll_dir)
            return

        now = time.time()
        if self._boundary_hit == 'top' and scroll_dir < 0 and now > self._boundary_hit_time:
            self.check_for_image_navigation(scroll_dir)
            return
        if self._boundary_hit == 'bottom' and scroll_dir > 0 and now > self._boundary_hit_time:
            self.check_for_image_navigation(scroll_dir)
            return
        
        self.canvas.yview_scroll(scroll_dir, "units")
        self.after(10, self.update_boundary_state)
        
    def update_boundary_state(self):
        if not self.winfo_exists(): return
        new_pos = self.v_scroll.get()
        at_top, at_bottom = new_pos[0] == 0.0, new_pos[1] == 1.0
        
        if at_top and self._boundary_hit != 'top':
            self._boundary_hit = 'top'
            self._boundary_hit_time = time.time() + 0.3
        elif at_bottom and self._boundary_hit != 'bottom':
            self._boundary_hit = 'bottom'
            self._boundary_hit_time = time.time() + 0.3
        elif not at_top and not at_bottom:
            self._boundary_hit = None
    
    def check_for_image_navigation(self, scroll_dir):
        if self._is_navigating or not self.app.current_image_path or not self.app.image_files: return
        current_file = os.path.basename(self.app.current_image_path)
        if not current_file in self.app.image_files: return
            
        current_index = self.app.image_files.index(current_file)
        next_index = -1
        if scroll_dir > 0 and current_index < len(self.app.image_files) - 1: next_index = current_index + 1
        elif scroll_dir < 0 and current_index > 0: next_index = current_index - 1
            
        if next_index != -1:
            self._is_navigating = True
            self.app.display_image(self.app.image_files[next_index])
            self.after(500, lambda: setattr(self, '_is_navigating', False))

    def set_zoom(self, zoom_ratio):
        self.zoom_ratio = zoom_ratio
        if self.original_image: self.prepare_display_images()

    def update_images(self, original_pil, translated_pil, is_new_folder=False):
        if original_pil is None: return
        self.original_image, self.translated_image = original_pil, translated_pil
        self._boundary_hit = None
        self.prepare_display_images(is_new_folder=is_new_folder)

    def prepare_display_images(self, is_new_folder=False):
        if not self.original_image: return
        canvas_w, canvas_h = self.winfo_width(), self.winfo_height()
        if canvas_w < 50 or canvas_h < 50: self.after(50, self.prepare_display_images); return
        self.fit_ratio = min(canvas_w / self.original_image.width, canvas_h / self.original_image.height) if canvas_w > 1 and self.original_image.width > 1 else 1.0
        
        if is_new_folder: self.zoom_ratio = self.fit_ratio
        
        self.master.update_zoom_entry(self.zoom_ratio, self.fit_ratio)
        self.canvas.delete("all")
        new_w, new_h = int(self.original_image.width * self.zoom_ratio), int(self.original_image.height * self.zoom_ratio)
        
        if self.performance_mode_var.get():
            resample = Image.Resampling.NEAREST
        else:
            resample = Image.Resampling.BILINEAR
        
        if new_w > 0 and new_h > 0:
            self.disp_orig = self.original_image.resize((new_w, new_h), resample)
            self.disp_trans = self.translated_image.resize((new_w, new_h), resample) if self.translated_image else self.disp_orig
        else:
             self.disp_orig, self.disp_trans = self.original_image, self.translated_image or self.disp_orig

        self.draw_image(self.slider.get())
        self.update_layout()

    def scroll_to_top(self):
        self.canvas.yview_moveto(0.0)

    def draw_image(self, slider_value):
        if not self.disp_orig: return
        if self.image_id: self.canvas.delete(self.image_id)
        new_w, new_h = self.disp_orig.size
        
        if self.disp_orig is self.disp_trans: composite_img = self.disp_orig
        elif not self.live_preview_var.get(): composite_img = self.disp_trans
        else:
            composite_img = self.disp_orig.copy()
            split_x = int(new_w * slider_value)
            if split_x > 0 and self.disp_trans:
                composite_img.paste(self.disp_trans.crop((0, 0, split_x, new_h)), (0, 0))

        self.photo_image_ref = ImageTk.PhotoImage(composite_img)
        img_start_x, img_start_y = self._get_image_start_coords()
        self.image_id = self.canvas.create_image(img_start_x, img_start_y, anchor="nw", image=self.photo_image_ref)
        self.canvas.config(scrollregion=(img_start_x, img_start_y, img_start_x + new_w, img_start_y + new_h))
        self.update_divider_line(slider_value)

    def update_divider_line(self, slider_value):
        if self.divider_id: self.canvas.delete(self.divider_id)
        if not self.live_preview_var.get() or self.disp_orig is self.disp_trans: self.divider_id = None; return

        img_w, img_h = self.disp_orig.size
        canvas_w = self.canvas.winfo_width()
        img_start_x, img_start_y = self._get_image_start_coords()
        visible_width = min(img_w, canvas_w)

        button_length = ctk.ThemeManager.theme["CTkSlider"].get("button_length", 16)
        split_x_on_canvas = int(((slider_value * (visible_width - button_length)) + (button_length / 2)))

        accent_color_data = ctk.ThemeManager.theme["CTkButton"]["fg_color"]
        accent_color = (accent_color_data[0] if ctk.get_appearance_mode() == "Dark" else accent_color_data[1]) if isinstance(accent_color_data, (list, tuple)) else accent_color_data

        self.divider_id = self.canvas.create_line(img_start_x + split_x_on_canvas, img_start_y, img_start_x + split_x_on_canvas, img_start_y + img_h, fill=accent_color, width=3, dash=(6, 4))
        
    def _get_image_start_coords(self):
        canvas_w, canvas_h = self.winfo_width(), self.winfo_height()
        if not self.disp_orig: return (0,0)
        img_w, img_h = self.disp_orig.size
        return (canvas_w - img_w) // 2 if img_w < canvas_w else 0, (canvas_h - img_h) // 2 if img_h < canvas_h else 0

    def update_layout(self):
        if not self.disp_orig: return
        canvas_w, canvas_h, img_w, img_h = self.winfo_width(), self.winfo_height(), self.disp_orig.width, self.disp_orig.height
        self.canvas.config(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        self.v_scroll.grid(row=0, column=1, sticky="ns") if img_h > canvas_h else self.v_scroll.grid_remove()
        self.h_scroll.grid(row=1, column=0, sticky="ew") if img_w > canvas_w else self.h_scroll.grid_remove()
        
        if self.disp_orig is not self.disp_trans and self.live_preview_var.get():
            img_start_x, _ = self._get_image_start_coords()
            slider_rel_w = min(img_w, canvas_w) / canvas_w if canvas_w > 0 else 0
            self.slider.place(relx=(img_start_x / canvas_w if canvas_w > 0 else 0), rely=0.98, relwidth=slider_rel_w, anchor="sw")
        else: self.slider.place_forget()

    def on_slider_move(self, value):
        if self.live_preview_var.get(): self.draw_image(value)
        else: self.update_divider_line(value)

    def on_slider_release(self, event):
        if not self.live_preview_var.get(): self.draw_image(self.slider.get())

    def on_resize(self, event):
        if self._after_id: self.after_cancel(self._after_id)
        if self.original_image: self._after_id = self.after(250, self.prepare_display_images)

# --- Theme Editor Window ---
class ThemeEditorWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.master_app = master
        self.title("Theme & Font Editor")
        self.geometry("400x520")
        self.transient(master); self.grab_set()
        self.grid_columnconfigure(0, weight=1)
        
        appearance_frame = ctk.CTkFrame(self, fg_color="transparent")
        appearance_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        appearance_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(appearance_frame, text="Appearance Mode:").grid(row=0, column=0, sticky="w")
        self.appearance_menu = ctk.CTkOptionMenu(appearance_frame, values=["Dark", "Light", "System"], command=self.change_appearance)
        self.appearance_menu.set(ctk.get_appearance_mode())
        self.appearance_menu.grid(row=0, column=1, padx=10, sticky="ew")

        hex_frame = ctk.CTkFrame(self, fg_color="transparent")
        hex_frame.grid(row=1, column=0, padx=20, pady=(5,10), sticky="ew")
        hex_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(hex_frame, text="Custom Hex Color:").pack(anchor="w")
        self.hex_entry = ctk.CTkEntry(hex_frame, placeholder_text="#RRGGBB or #RGB")
        self.hex_entry.pack(fill="x", pady=(0,5))
        self.hex_entry.bind("<Return>", self.apply_custom_color_event)
        
        ctk.CTkLabel(self, text="Or Choose a Preset:").grid(row=2, column=0, padx=20, sticky="w")
        self.preset_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.preset_frame.grid(row=3, column=0, padx=20, pady=5, sticky="ew")
        
        self.colors = {"Blue": "#3B8ED0", "Green": "#2CC985", "Red": "#D03B3B", "Orange": "#D08F3B", "Purple": "#8A3BD0", "Pink": "#D03B9C"}
        for i, (name, code) in enumerate(self.colors.items()):
            col, row = i % 3, i // 3
            self.preset_frame.grid_columnconfigure(col, weight=1)
            btn = ctk.CTkButton(self.preset_frame, text=name, fg_color=code, hover_color=self.adjust_color(code, -0.2), text_color=self.get_text_color_for_bg(code), command=lambda c=code: self.apply_custom_color(c))
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")

        font_size_backup = self.master_app.font_size_var.get()
        self.master_app.font_size_var = tk.StringVar(value=font_size_backup)

        font_frame = ctk.CTkFrame(self, fg_color="transparent")
        font_frame.grid(row=4, column=0, padx=20, pady=15, sticky="ew")
        font_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(font_frame, text="Translation Font:").grid(row=0, column=0, padx=(0,10), sticky="w")
        system_fonts = sorted([f for f in font.families() if not f.startswith('@')])
        self.font_menu = ctk.CTkOptionMenu(font_frame, values=system_fonts, variable=self.master_app.font_family_var)
        self.font_menu.grid(row=0, column=1, sticky="ew")

        ctk.CTkLabel(font_frame, text="Font Size:").grid(row=1, column=0, pady=(10,0), padx=(0,10), sticky="w")
        self.font_size_entry = ctk.CTkEntry(font_frame, placeholder_text="e.g., 48 (blank for auto)", textvariable=self.master_app.font_size_var)
        self.font_size_entry.grid(row=1, column=1, pady=(10,0), sticky="ew")
        
        warning_label = ctk.CTkLabel(font_frame, 
                                     text="Note: Custom fonts/sizes may not fit perfectly in all text bubbles.",
                                     text_color="gray",
                                     font=ctk.CTkFont(size=12),
                                     wraplength=300,
                                     justify="left")
        warning_label.grid(row=2, column=0, columnspan=2, pady=(8, 0), padx=5, sticky="w")
        
        self.apply_button = ctk.CTkButton(self, text="Apply Font & Retranslate", command=self.apply_font_and_retranslate)
        self.apply_button.grid(row=5, column=0, padx=20, pady=(20,15), sticky="ew")

    def apply_font_and_retranslate(self):
        self.master_app.start_translation()

    def change_appearance(self, new_mode):
        ctk.set_appearance_mode(new_mode)
        self.master_app.refresh_ui()

    def get_text_color_for_bg(self, bg_color: str):
        try: bg = bg_color.lstrip('#'); r,g,b = tuple(int(bg[i:i+2], 16) for i in (0,2,4)); return "#000" if (r*0.299+g*0.587+b*0.114)>149 else "#FFF"
        except: return "#FFF"

    def apply_custom_color_event(self, event):
        self.apply_custom_color(self.hex_entry.get())

    def adjust_color(self, c_hex, factor):
        try:
            c_hex = c_hex.lstrip('#'); r,g,b = tuple(int(c_hex[i:i+2],16) for i in (0,2,4))
            r,g,b = int(max(0,min(255,r*(1+factor)))), int(max(0,min(255,g*(1+factor)))), int(max(0,min(255,b*(1+factor))))
            return f"#{r:02x}{g:02x}{b:02x}"
        except: return c_hex
    
    def recursive_replace(self, data, replacements):
        if isinstance(data, dict): return {k: self.recursive_replace(v, replacements) for k, v in data.items()}
        elif isinstance(data, list) and len(data) == 2 and all(isinstance(i, str) and i.startswith("#") for i in data):
            return [replacements.get(data[0].upper(), data[0]), replacements.get(data[1].upper(), data[1])]
        elif isinstance(data, list): return [self.recursive_replace(item, replacements) for item in data]
        elif isinstance(data, str): return replacements.get(data.upper(), data)
        return data

    def apply_custom_color(self, color_hex: str):
        if not re.match(r"^#(?:[0-9a-fA-F]{3}){1,2}$", color_hex): self.hex_entry.delete(0, "end"); self.hex_entry.insert(0, "Invalid Hex"); return
        if len(color_hex) == 4: color_hex = f"#{color_hex[1]*2}{color_hex[2]*2}{color_hex[3]*2}"
        theme_path = os.path.join(os.path.dirname(ctk.__file__), "assets", "themes", "blue.json")
        with open(theme_path, "r") as f: base_theme = json.load(f)
        
        replacements = {"#3B8ED0": color_hex, "#36719F": self.adjust_color(color_hex, -0.15), "#1F6AA5": color_hex, "#144870": self.adjust_color(color_hex, -0.15)}
        final_theme = self.recursive_replace(base_theme, replacements)
        new_text_color = self.get_text_color_for_bg(color_hex)
        
        if "CTkButton" in final_theme: final_theme["CTkButton"]["text_color"] = [new_text_color, new_text_color]
        if "CTkOptionMenu" in final_theme: final_theme["CTkOptionMenu"]["text_color"] = [new_text_color, new_text_color]

        with open("custom_theme.json", "w") as f: json.dump(final_theme, f, indent=4)
        ctk.set_default_color_theme("custom_theme.json")
        self.master_app.apply_new_theme_live()
        self.update_editor_theme()

    def update_editor_theme(self):
        try:
            theme = ctk.ThemeManager.theme
            accent_color, hover_color, text_color = theme["CTkButton"]["fg_color"], theme["CTkButton"]["hover_color"], theme["CTkButton"]["text_color"]
            self.apply_button.configure(fg_color=accent_color, hover_color=hover_color, text_color=text_color)
            option_theme = theme["CTkOptionMenu"]
            self.appearance_menu.configure(fg_color=option_theme["fg_color"], button_color=option_theme["button_color"], button_hover_color=option_theme["button_hover_color"], text_color=option_theme["text_color"])
            self.font_menu.configure(fg_color=option_theme["fg_color"], button_color=option_theme["button_color"], button_hover_color=option_theme["button_hover_color"], text_color=option_theme["text_color"])
        except Exception as e:
            print(f"Error updating editor theme: {e}")

class ManhwaTranslatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Manhwa Translator v28")
        self.geometry("1200x750")
        self.grid_columnconfigure(0, weight=3); self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)
        ctk.set_appearance_mode("Dark"); ctk.set_default_color_theme("blue")
        self.input_folder, self.current_image_path, self.model, self.device = "", None, None, None
        self.image_files, self.file_labels, self.translation_cache = [], {}, {}
        self.live_preview_var = tk.BooleanVar(value=True); self.performance_mode_var = tk.BooleanVar(value=False)
        self.zoom_entry_var = tk.StringVar()
        self.font_family_var = tk.StringVar(value="Bangers-Regular.ttf")
        self.font_size_var = tk.StringVar(value="")
        self.theme_editor_window, self.image_viewer = None, None
        self.create_widgets()
        self.load_model_thread()

    def create_widgets(self):
        zoom = self.image_viewer.zoom_ratio if self.image_viewer else 1.0
        fit = self.image_viewer.fit_ratio if self.image_viewer else 1.0
        img_path, files = self.current_image_path, self.image_files
        font_family = self.font_family_var.get()
        font_size = self.font_size_var.get()

        for widget in self.winfo_children(): widget.destroy()

        self.font_family_var = tk.StringVar(value=font_family)
        self.font_size_var = tk.StringVar(value=font_size)
        self.zoom_entry_var = tk.StringVar() # Added to fix ghost widget on zoom entry

        self.image_viewer = BeforeAfterSlider(self, live_preview_var=self.live_preview_var, performance_mode_var=self.performance_mode_var)
        self.image_viewer.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        self.control_frame.grid_rowconfigure(6, weight=1); self.control_frame.grid_columnconfigure(0, weight=1)

        self.select_button = ctk.CTkButton(self.control_frame, text="Select Folder", command=self.select_folder, state="disabled")
        self.select_button.grid(row=0, column=0, padx=15, pady=(15, 7), sticky="ew")
        self.folder_path_label = ctk.CTkLabel(self.control_frame, text="AI Model is loading...", text_color="gray")
        self.folder_path_label.grid(row=1, column=0, padx=15, pady=(0, 7), sticky="w")
        
        options_frame = ctk.CTkFrame(self.control_frame); options_frame.grid(row=2, column=0, padx=15, pady=10, sticky="ew")
        options_frame.grid_columnconfigure(0, weight=1)
        self.theme_edit_button = ctk.CTkButton(options_frame, text="Edit Theme & Font...", command=self.open_theme_editor)
        self.theme_edit_button.pack(fill="x", padx=10, pady=10)
        
        zoom_frame = ctk.CTkFrame(self.control_frame); zoom_frame.grid(row=3, column=0, padx=15, pady=5, sticky="ew")
        ctk.CTkLabel(zoom_frame, text="Zoom:").pack(side="left", padx=(10, 5))
        self.zoom_entry = ctk.CTkEntry(zoom_frame, textvariable=self.zoom_entry_var, width=55)
        self.zoom_entry.pack(side="left"); self.zoom_entry.bind("<Return>", self.on_zoom_apply)
        ctk.CTkLabel(zoom_frame, text="%").pack(side="left", padx=(5,0))
        
        self.live_preview_checkbox = ctk.CTkCheckBox(self.control_frame, text="Live Slider Preview", variable=self.live_preview_var, command=self.on_live_preview_toggle)
        self.live_preview_checkbox.grid(row=4, column=0, padx=15, pady=5, sticky="w")
        self.performance_mode_checkbox = ctk.CTkCheckBox(self.control_frame, text="Performance Mode (Faster Scrolling)", variable=self.performance_mode_var, command=self.on_performance_mode_toggle)
        self.performance_mode_checkbox.grid(row=5, column=0, padx=15, pady=5, sticky="w")
        
        self.file_list_frame = ctk.CTkScrollableFrame(self.control_frame, label_text="Image Files")
        self.file_list_frame.grid(row=6, column=0, padx=15, pady=0, sticky="nsew")
        self.progress_bar = ctk.CTkProgressBar(self.control_frame); self.progress_bar.grid(row=7, column=0, padx=15, pady=15, sticky="ew")
        self.status_label = ctk.CTkLabel(self.control_frame, text="Please wait, loading AI model...", text_color="orange")
        self.status_label.grid(row=8, column=0, padx=15, pady=10, sticky="w")
        self.translate_button = ctk.CTkButton(self.control_frame, text="Translate Current Image!", command=self.start_translation, state="disabled")
        self.translate_button.grid(row=9, column=0, padx=15, pady=15, sticky="ew")
        
        self.current_image_path, self.image_files = img_path, files
        if self.image_viewer:
             self.image_viewer.zoom_ratio, self.image_viewer.fit_ratio = zoom, fit
             self.update_zoom_entry(zoom)
        if self.image_files: self.refresh_file_list()
        if self.current_image_path: self.display_image(os.path.basename(self.current_image_path))

    def on_theme_editor_close(self):
        if self.theme_editor_window: self.theme_editor_window.destroy(); self.theme_editor_window = None

    def open_theme_editor(self):
        if self.theme_editor_window is None or not self.theme_editor_window.winfo_exists():
            self.theme_editor_window = ThemeEditorWindow(self)
            self.theme_editor_window.protocol("WM_DELETE_WINDOW", self.on_theme_editor_close)
        else: self.theme_editor_window.focus()
    
    def refresh_ui(self):
        editor_open = self.theme_editor_window is not None and self.theme_editor_window.winfo_exists()
        geo = self.theme_editor_window.geometry() if editor_open else None
        self.create_widgets()
        if self.model: self.status_label.configure(text="AI Ready."), self.select_button.configure(state="normal")
        if self.input_folder: self.folder_path_label.configure(text=os.path.basename(self.input_folder))
        if self.image_viewer: self.image_viewer.update_canvas_bg()
        if editor_open: self.open_theme_editor(); self.theme_editor_window.geometry(geo) if geo else None

    def apply_new_theme_live(self):
        try:
            theme = ctk.ThemeManager.theme
            btn_fg, btn_hover = theme["CTkButton"]["fg_color"], theme["CTkButton"]["hover_color"]
            btn_text = theme["CTkButton"]["text_color"]
            chk_fg = theme["CTkCheckBox"]["fg_color"]
            prog_color = theme["CTkProgressBar"]["progress_color"]
            slider_btn, slider_hover = theme["CTkSlider"]["button_color"], theme["CTkSlider"]["button_hover_color"]
            
            self.select_button.configure(fg_color=btn_fg, hover_color=btn_hover, text_color=btn_text)
            self.translate_button.configure(fg_color=btn_fg, hover_color=btn_hover, text_color=btn_text)
            self.theme_edit_button.configure(fg_color=btn_fg, hover_color=btn_hover, text_color=btn_text)
            self.live_preview_checkbox.configure(fg_color=chk_fg); self.performance_mode_checkbox.configure(fg_color=chk_fg)
            self.progress_bar.configure(progress_color=prog_color)
            
            if self.image_viewer:
                self.image_viewer.slider.configure(button_color=slider_btn, button_hover_color=slider_hover)
                self.image_viewer.update_divider_line(self.image_viewer.slider.get())
            if self.current_image_path: self.highlight_selected_file(os.path.basename(self.current_image_path))
            if self.theme_editor_window and self.theme_editor_window.winfo_exists():
                self.theme_editor_window.update_editor_theme()
        except Exception as e:
            print(f"Error applying theme live, falling back: {e}"); self.refresh_ui()

    def on_zoom_apply(self, event=None):
        if not self.image_viewer: return
        try:
            val = int(self.zoom_entry_var.get().replace('%',''))
            val = max(int(self.image_viewer.fit_ratio*100), min(100, val))
            self.zoom_entry_var.set(str(val)); self.image_viewer.set_zoom(val/100.0)
        except: self.update_zoom_entry(self.image_viewer.zoom_ratio)

    def update_zoom_entry(self, zoom_ratio, fit_ratio=None):
        if self.image_viewer and fit_ratio: self.image_viewer.fit_ratio = fit_ratio
        self.zoom_entry_var.set(str(int(zoom_ratio * 100)))

    def on_live_preview_toggle(self):
        if self.image_viewer: self.image_viewer.draw_image(self.image_viewer.slider.get()); self.image_viewer.update_layout()

    def on_performance_mode_toggle(self):
        if self.image_viewer and self.image_viewer.original_image: self.image_viewer.prepare_display_images()

    def reorder_file(self, filename, new_index):
        if filename is not None and new_index is not None:
            self.image_files.remove(filename); self.image_files.insert(new_index, filename)
        self.refresh_file_list()

    def refresh_file_list(self):
        for widget in self.file_list_frame.winfo_children(): widget.destroy()
        self.file_labels = {} 
        for filename in self.image_files:
            item = DraggableFileItem(self.file_list_frame, filename=filename, reorder_callback=self.reorder_file, select_callback=self.display_image)
            item.pack(fill="x", padx=5, pady=2); self.file_labels[filename] = item
        if self.current_image_path: self.highlight_selected_file(os.path.basename(self.current_image_path))

    def load_model_thread(self):
        if not get_model: self.status_label.configure(text="Error: processing.py not found.", text_color="red"); return
        self.progress_bar.start()
        threading.Thread(target=self.load_model_logic, daemon=True).start()

    def load_model_logic(self):
        self.model, self.device = get_model()
        self.after(0, self.on_model_loaded)

    def on_model_loaded(self):
        self.progress_bar.stop(); self.progress_bar.set(1)
        if self.model: self.status_label.configure(text="AI Ready."), self.select_button.configure(state="normal")
        else: self.status_label.configure(text="CRITICAL: AI Model failed.", text_color="red")

    def select_folder(self):
        folder = filedialog.askdirectory()
        if not folder: return
        self.input_folder, self.translation_cache = folder, {}
        self.folder_path_label.configure(text=os.path.basename(folder))
        self.image_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if self.image_files:
            self.status_label.configure(text=f"{len(self.image_files)} images found.")
            self.refresh_file_list(); self.display_image(self.image_files[0], new_folder=True)
        else: self.status_label.configure(text="No images found.")
    
    def highlight_selected_file(self, filename_to_select):
        for name, item_frame in self.file_labels.items():
            item_frame.configure_appearance(is_selected=(name == filename_to_select))

    def display_image(self, filename, new_folder=False):
        self.current_image_path = os.path.join(self.input_folder, filename)
        try: original_pil = Image.open(self.current_image_path).convert("RGB")
        except Exception as e: self.status_label.configure(text=f"Error opening {filename}", text_color="red"); return
        
        if self.image_viewer:
            translated_pil = self.translation_cache.get(filename)
            self.image_viewer.update_images(original_pil, translated_pil, is_new_folder=new_folder)
            self.highlight_selected_file(filename)
            self.translate_button.configure(state="normal")
            self.status_label.configure(text=f"Viewing: {filename}")
            self.image_viewer.scroll_to_top()

    def start_translation(self):
        if not self.current_image_path: return
        self.status_label.configure(text="Translating, please wait...")
        self.translate_button.configure(state="disabled"); self.select_button.configure(state="disabled")
        self.progress_bar.start()
        
        font_family = self.font_family_var.get()
        font_size_str = self.font_size_var.get()
        try: font_size = int(font_size_str) if font_size_str else None
        except ValueError: font_size = None 
        
        threading.Thread(target=self.translation_logic, args=(font_family, font_size), daemon=True).start()

    def translation_logic(self, font_family, font_size):
        translated_image = process_single_image(self.current_image_path, self.model, self.device, font_family, font_size)
        if translated_image:
            self.translation_cache[os.path.basename(self.current_image_path)] = translated_image.convert("RGB")
        self.after(0, self.translation_done, translated_image is not None)

    def translation_done(self, success):
        self.progress_bar.stop(); self.progress_bar.set(1)
        if success:
            self.status_label.configure(text="Translation complete!", text_color="light green")
            self.display_image(os.path.basename(self.current_image_path))
        else:
            self.status_label.configure(text="Translation failed. Check console.", text_color="red")
        self.select_button.configure(state="normal"); self.translate_button.configure(state="normal")

if __name__ == "__main__":
    if process_single_image is None:
        root = tk.Tk(); root.withdraw()
        from tkinter import messagebox
        messagebox.showerror("Fatal Error", "processing.py not found. Make sure it's in the same folder as app.py")
    else:
        app = ManhwaTranslatorApp()
        app.mainloop()