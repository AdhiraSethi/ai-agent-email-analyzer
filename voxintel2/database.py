"""
database.py — VoxIntel persistent storage
SQLite for dev, swap connection string for PostgreSQL in production.
"""

import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Float,
    Integer, Boolean, DateTime, Text, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from contextlib import contextmanager

DB_URL = os.getenv("DATABASE_URL", "sqlite:///voxintel.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class EmailLog(Base):
    __tablename__ = "email_logs"
    id              = Column(String,  primary_key=True)
    customer_id     = Column(String,  index=True)
    sender          = Column(String)
    subject         = Column(String)
    body            = Column(Text)
    summary         = Column(Text)
    intent          = Column(String)
    emotion         = Column(String)
    sentiment       = Column(String)
    priority        = Column(String)
    urgency         = Column(String)
    department      = Column(String)
    language        = Column(String)
    risk_flagged    = Column(Boolean, default=False)
    customer_value  = Column(String)
    confidence      = Column(Float)
    decision        = Column(String)
    decision_reason = Column(String)
    ai_reply        = Column(Text)
    outgoing_email  = Column(JSON)
    manager_alert   = Column(JSON)
    processing_ms   = Column(Float)
    created_at      = Column(DateTime, default=datetime.utcnow)


class Ticket(Base):
    __tablename__ = "tickets"
    id          = Column(String,  primary_key=True)
    email_id    = Column(String)
    customer_id = Column(String,  index=True)
    sender      = Column(String)
    subject     = Column(String)
    intent      = Column(String)
    priority    = Column(String)
    status      = Column(String,  default="open")
    assigned_to = Column(String,  default="unassigned")
    department  = Column(String)
    notes       = Column(Text,    default="")
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)


class Customer(Base):
    __tablename__ = "customers"
    id              = Column(String,  primary_key=True)
    email           = Column(String,  unique=True, index=True)
    name            = Column(String,  default="Customer")
    lifetime_value  = Column(Float,   default=0.0)
    complaint_count = Column(Integer, default=0)
    ticket_count    = Column(Integer, default=0)
    is_vip          = Column(Boolean, default=False)
    tenure_months   = Column(Integer, default=0)
    last_seen       = Column(DateTime, default=datetime.utcnow)
    created_at      = Column(DateTime, default=datetime.utcnow)


class Correction(Base):
    __tablename__ = "corrections"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    email_id    = Column(String)
    ai_reply    = Column(Text)
    human_reply = Column(Text)
    changed     = Column(Boolean)
    created_at  = Column(DateTime, default=datetime.utcnow)


class Memory(Base):
    __tablename__ = "memory"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String, index=True)
    summary     = Column(Text)
    resolved    = Column(Boolean, default=False)
    created_at  = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(engine)


@contextmanager
def get_db():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_email_log(session, email_id, req, result, elapsed):
    analysis = result["analysis"]
    session.add(EmailLog(
        id              = email_id,
        customer_id     = req.get("customer_id", ""),
        sender          = req.get("sender", ""),
        subject         = req.get("subject", ""),
        body            = req.get("body", "")[:2000],
        summary         = analysis.get("summary", ""),
        intent          = analysis.get("intent", ""),
        emotion         = analysis.get("emotion", ""),
        sentiment       = analysis.get("sentiment", ""),
        priority        = analysis.get("priority", ""),
        urgency         = analysis.get("urgency", ""),
        department      = analysis.get("department", ""),
        language        = analysis.get("language", "en"),
        risk_flagged    = analysis.get("risk_flagged", False),
        customer_value  = analysis.get("customer_value", "Regular"),
        confidence      = result["reply"]["confidence"],
        decision        = result["decision"],
        decision_reason = result["decision_reason"],
        ai_reply        = result["reply"]["reply"],
        outgoing_email  = result.get("outgoing_email"),
        manager_alert   = result.get("manager_alert"),
        processing_ms   = elapsed,
    ))


def create_ticket(session, email_id, req, result):
    count     = session.query(Ticket).count() + 1
    ticket_id = f"TICKET-{count:04d}"
    analysis  = result["analysis"]
    session.add(Ticket(
        id          = ticket_id,
        email_id    = email_id,
        customer_id = req.get("customer_id", ""),
        sender      = req.get("sender", ""),
        subject     = req.get("subject", ""),
        intent      = analysis.get("intent", ""),
        priority    = analysis.get("priority", "Medium"),
        department  = analysis.get("department", "Customer Care"),
    ))
    return ticket_id


def upsert_customer(session, customer_id, sender, name):
    customer = session.query(Customer).filter_by(email=sender).first()
    if not customer:
        session.add(Customer(id=customer_id or sender, email=sender, name=name))
    else:
        customer.ticket_count += 1
        customer.last_seen = datetime.utcnow()


def save_memory(session, customer_id, summary, resolved):
    if customer_id:
        session.add(Memory(customer_id=customer_id, summary=summary, resolved=resolved))


def get_memory(session, customer_id, limit=5):
    rows = (
        session.query(Memory)
        .filter_by(customer_id=customer_id)
        .order_by(Memory.created_at.desc())
        .limit(limit).all()
    )
    return [
        {"time": r.created_at.strftime("%Y-%m-%d %H:%M"), "summary": r.summary, "resolved": r.resolved}
        for r in reversed(rows)
    ]


def save_correction(session, email_id, ai_reply, human_reply):
    session.add(Correction(
        email_id    = email_id,
        ai_reply    = ai_reply,
        human_reply = human_reply,
        changed     = ai_reply.strip() != human_reply.strip(),
    ))