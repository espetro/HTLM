import { beforeEach, describe, expect, it } from "vitest";
import { extractPage, isVisible, labelOf, roleOf } from "~/lib/content-map";

function setBody(html: string) {
  document.body.innerHTML = html;
}

// jsdom has no layout engine, so getBoundingClientRect() is always a zero rect. Stub it with a
// fixed non-zero size so isVisible()'s width/height guard doesn't mask the style-based checks
// (display/visibility/opacity) these tests actually exercise.
beforeEach(() => {
  Element.prototype.getBoundingClientRect = () =>
    ({ x: 0, y: 0, width: 100, height: 20, top: 0, left: 0, right: 100, bottom: 20, toJSON() {} }) as DOMRect;
});

describe("roleOf", () => {
  it("maps ARIA role over tag", () => {
    const el = document.createElement("div");
    el.setAttribute("role", "tab");
    expect(roleOf(el)).toBe("tab");
  });

  it("maps <a href> to link, <a> without href to null", () => {
    const withHref = document.createElement("a");
    withHref.setAttribute("href", "/x");
    expect(roleOf(withHref)).toBe("link");
    const withoutHref = document.createElement("a");
    expect(roleOf(withoutHref)).toBeNull();
  });

  it("maps input types", () => {
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    expect(roleOf(checkbox)).toBe("checkbox");
    const hidden = document.createElement("input");
    hidden.type = "hidden";
    expect(roleOf(hidden)).toBeNull();
    const text = document.createElement("input");
    expect(roleOf(text)).toBe("input");
  });

  it("maps [role=combobox]", () => {
    const el = document.createElement("div");
    el.setAttribute("role", "combobox");
    expect(roleOf(el)).toBe("combobox");
  });
});

describe("labelOf", () => {
  it("prefers aria-label", () => {
    const el = document.createElement("button");
    el.setAttribute("aria-label", "Submit form");
    el.textContent = "ignored";
    expect(labelOf(el)).toBe("Submit form");
  });

  it("falls back to text content", () => {
    const el = document.createElement("button");
    el.textContent = "  Click me  ";
    expect(labelOf(el)).toBe("Click me");
  });
});

describe("isVisible", () => {
  it("returns false for display:none elements", () => {
    setBody(`<div id="hidden" style="display:none">x</div>`);
    const el = document.getElementById("hidden")!;
    expect(isVisible(el)).toBe(false);
  });
});

describe("extractPage", () => {
  beforeEach(() => {
    setBody(`
      <a href="/go" aria-label="Go home">Home</a>
      <button>Submit</button>
      <input type="text" placeholder="Search" />
      <div role="tab" aria-label="Tab one">Tab 1</div>
      <div id="hidden" style="display:none" role="button">hidden button</div>
      <div role="combobox" aria-label="Pick one">Combo</div>
    `);
  });

  it("extracts roles/labels, skips hidden elements", () => {
    const { page, elements } = extractPage();
    const roles = page.elements.map((e) => e.role);
    expect(roles).toEqual(["link", "button", "input", "tab", "combobox"]);
    expect(page.elements[0].label).toBe("Go home");
    expect(elements).toHaveLength(page.elements.length);
    expect(elements[0].tagName.toLowerCase()).toBe("a");
  });

  it("assigns sequential indices aligned with the live element map", () => {
    const { page, elements } = extractPage();
    page.elements.forEach((el, i) => {
      expect(el.index).toBe(i);
      expect(elements[i]).toBeInstanceOf(Element);
    });
  });

  it("caps at MAX_ELEMENTS (64)", () => {
    const many = Array.from({ length: 80 }, (_, i) => `<button>btn ${i}</button>`).join("");
    setBody(many);
    const { page } = extractPage();
    expect(page.elements).toHaveLength(64);
  });
});
