"""Offline MIME extraction; retain visible text and hidden HTML destinations.

URLs are data only: no links, images or attachments are fetched.
The body stays separate from model_body so UI text is not silently rewritten.
"""
from email.parser import BytesParser
from email.policy import default
from html.parser import HTMLParser
from urllib.parse import urlsplit
import mailbox
import re


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.links, self.images = [], [], 0
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('script', 'style', 'head'):
            self.hidden += 1
        if self.hidden:
            return
        if tag in ('p', 'div', 'br', 'li', 'tr', 'td', 'h1', 'h2', 'h3'):
            self.parts.append('\n')
        if tag == 'a':
            href = attrs.get('href', '').strip()
            try:
                p = urlsplit(href)
                valid = p.scheme.lower() in ('http', 'https') and bool(p.hostname)
            except ValueError:
                valid = False
            if valid and not any(ord(c) < 32 for c in href):
                self.links.append(href)
        if tag == 'img':
            self.images += 1
            if attrs.get('alt'):
                self.parts.append(attrs['alt'])

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'head'):
            self.hidden = max(0, self.hidden - 1)
        if not self.hidden and tag in ('p', 'div', 'li', 'tr'):
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def readable_message(raw_bytes):
    message = BytesParser(policy=default).parsebytes(raw_bytes)
    plain, html, links = [], [], []
    image_count, attachment_count = 0, 0

    def collect(part):
        nonlocal image_count, attachment_count
        # Skip attachment subtrees, including attached messages and multiparts.
        if part.get_content_disposition() == 'attachment' or part.get_filename():
            attachment_count += 1
            return
        if part.get_content_type() == 'message/rfc822':
            return
        if part.is_multipart():
            for child in part.iter_parts():
                collect(child)
            return
        kind = part.get_content_type()
        if kind not in ('text/plain', 'text/html'):
            return
        try:
            text = part.get_content()
        except (LookupError, UnicodeError, KeyError):
            text = (part.get_payload(decode=True) or b'').decode('utf-8', errors='replace')
        if kind == 'text/plain':
            plain.append(text)
        else:
            parser = VisibleText()
            parser.feed(text)
            html.append(' '.join(parser.parts))
            links.extend(parser.links)
            image_count += parser.images

    collect(message)
    plain_text, html_text = '\n'.join(plain).strip(), '\n'.join(html).strip()
    placeholder = len(plain_text.split()) <= 20 and len(html_text.split()) >= max(20, 3 * len(plain_text.split()))
    body = html_text if placeholder else plain_text or html_text
    links = list(dict.fromkeys(links))
    visible_urls = set(re.findall(r'https?://[^\s<>"\']+', body))
    omitted = [url for url in links if url not in visible_urls]
    model_body = body
    if omitted:
        model_body += '\nHTML link destinations:\n' + '\n'.join(omitted)
    return dict(subject=str(message.get('Subject', '')), body=body,
                model_body=model_body, link_destinations=links,
                omitted_destination_count=len(omitted), image_count=image_count,
                attachment_count=attachment_count, has_plain=bool(plain),
                has_html=bool(html), selected_html=bool(html_text and (placeholder or not plain_text)),
                replaced_short_plain=bool(plain_text and html_text and placeholder))


def read_mailbox(path):
    box = mailbox.mbox(path)
    try:
        for key in box.iterkeys():
            yield readable_message(box.get_bytes(key))
    finally:
        box.close()
