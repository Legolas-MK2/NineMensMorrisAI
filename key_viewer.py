"""Simple key viewer — shows pressed keys in a window.

Requires: pygame  (pip install pygame)
Run: python key_viewer.py
"""

import pygame
import sys
from collections import deque


def main():
    pygame.init()
    width, height = 800, 400
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Key Viewer — press any key (ESC to quit)")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont("consolas", 72, bold=True)
    font_small = pygame.font.SysFont("consolas", 24)

    currently_held = set()
    history = deque(maxlen=10)

    bg = (20, 20, 28)
    fg = (230, 230, 240)
    accent = (120, 200, 255)
    dim = (120, 120, 140)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                name = pygame.key.name(event.key)
                currently_held.add(name)
                mods = []
                m = event.mod
                if m & pygame.KMOD_CTRL:
                    mods.append("Ctrl")
                if m & pygame.KMOD_ALT:
                    mods.append("Alt")
                if m & pygame.KMOD_SHIFT:
                    mods.append("Shift")
                if m & pygame.KMOD_META:
                    mods.append("Meta")
                combo = "+".join(mods + [name]) if mods else name
                history.appendleft(f"{combo}  (code={event.key})")
            elif event.type == pygame.KEYUP:
                currently_held.discard(pygame.key.name(event.key))

        pressed = pygame.key.get_pressed()
        polled_down = []
        for kcode in range(len(pressed)):
            if pressed[kcode]:
                polled_down.append(f"{pygame.key.name(kcode)} (code={kcode})")

        mods_now = pygame.key.get_mods()
        mod_list = []
        if mods_now & pygame.KMOD_CTRL:
            mod_list.append("Ctrl")
        if mods_now & pygame.KMOD_ALT:
            mod_list.append("Alt")
        if mods_now & pygame.KMOD_SHIFT:
            mod_list.append("Shift")
        if mods_now & pygame.KMOD_META:
            mod_list.append("Meta")

        screen.fill(bg)

        label = font_small.render("Polled (OS says DOWN right now):", True, dim)
        screen.blit(label, (20, 20))

        if polled_down:
            for i, entry in enumerate(polled_down[:6]):
                surf = font_small.render(entry, True, (255, 120, 120))
                screen.blit(surf, (40, 50 + i * 26))
        else:
            surf = font_small.render("(nothing — no stuck keys detected)", True, dim)
            screen.blit(surf, (40, 50))

        mod_text = "Modifiers: " + (" + ".join(mod_list) if mod_list else "—")
        surf = font_small.render(mod_text, True, accent)
        screen.blit(surf, (20, 220))

        held_text = " + ".join(sorted(currently_held)) if currently_held else "—"
        surf = font_small.render("Event-based held: " + held_text, True, fg)
        screen.blit(surf, (20, 250))

        hist_label = font_small.render("Event history:", True, dim)
        screen.blit(hist_label, (20, 290))

        for i, entry in enumerate(list(history)[:4]):
            color = fg if i == 0 else dim
            surf = font_small.render(entry, True, color)
            screen.blit(surf, (40, 315 + i * 22))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
