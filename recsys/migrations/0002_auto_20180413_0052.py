# Generated by Django 2.0.2 on 2018-04-13 00:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('recsys', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='post',
            name='description',
            field=models.CharField(max_length=200, null=True),
        ),
    ]
