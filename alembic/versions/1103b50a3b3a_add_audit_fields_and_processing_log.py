"""add audit fields and processing log

Revision ID: 1103b50a3b3a
Revises: 5e15395b2266
Create Date: 2026-03-10 20:00:28.763611

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1103b50a3b3a'
down_revision: Union[str, Sequence[str], None] = '5e15395b2266'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # -- New columns on documents table --
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.add_column(sa.Column('updated_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('error_category', sa.String(length=50), nullable=True))
        batch_op.add_column(sa.Column('field_confidence_json', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('processing_route', sa.String(length=30), nullable=True))
        batch_op.add_column(sa.Column('vendor_match_score', sa.Float(), nullable=True))
        batch_op.create_index(batch_op.f('ix_documents_error_category'), ['error_category'], unique=False)

    # -- New table: document_processing_logs --
    op.create_table('document_processing_logs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.Column('action', sa.String(length=50), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('error_category', sa.String(length=50), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('details_json', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('document_processing_logs', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_document_processing_logs_document_id'), ['document_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_document_processing_logs_id'), ['id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    # -- Drop document_processing_logs table --
    with op.batch_alter_table('document_processing_logs', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_document_processing_logs_id'))
        batch_op.drop_index(batch_op.f('ix_document_processing_logs_document_id'))

    op.drop_table('document_processing_logs')

    # -- Remove added columns from documents --
    with op.batch_alter_table('documents', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_documents_error_category'))
        batch_op.drop_column('vendor_match_score')
        batch_op.drop_column('processing_route')
        batch_op.drop_column('field_confidence_json')
        batch_op.drop_column('error_category')
        batch_op.drop_column('updated_at')
