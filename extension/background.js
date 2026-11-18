// Adds a right-click menu item: "Check for job scam".
// The selected text is stored and a badge is shown; clicking the toolbar
// icon then opens the popup with the text pre-filled.

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "check-job-scam",
    title: "Check selected text for job scam",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener(async (info) => {
  if (info.menuItemId !== "check-job-scam" || !info.selectionText) return;
  await chrome.storage.local.set({ pendingText: info.selectionText });
  chrome.action.setBadgeText({ text: "1" });
  chrome.action.setBadgeBackgroundColor({ color: "#dc2626" });
  // Chrome 127+ can open the popup programmatically; ignore if unsupported.
  try { await chrome.action.openPopup(); } catch (e) { /* user clicks the icon */ }
});
