/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.cassandra.cql3.validation.entities;

import java.util.Arrays;

import org.junit.Test;

import org.apache.cassandra.cql3.CQLTester;
import org.apache.cassandra.cql3.UntypedResultSet;
import org.apache.cassandra.db.marshal.Int32Type;
import org.apache.cassandra.db.marshal.ListType;
import org.apache.cassandra.exceptions.InvalidRequestException;

import static java.lang.String.format;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

public class WritetimeOrTTLTest extends CQLTester
{
    private static final long TIMESTAMP_1 = 1;
    private static final long TIMESTAMP_2 = 2;
    private static final Long NO_TIMESTAMP = null;

    private static final int TTL_1 = 10000;
    private static final int TTL_2 = 20000;
    private static final Integer NO_TTL = null;

    @Test
    public void testSimple() throws Throwable
    {
        createTable("CREATE TABLE %s (pk int, ck int, v int,  PRIMARY KEY(pk, ck))");

        // Primary key columns should be rejected
        assertInvalidPrimaryKeySelection("pk");
        assertInvalidPrimaryKeySelection("ck");

        // No rows
        assertEmpty(execute("SELECT WRITETIME(v) FROM %s"));
        assertEmpty(execute("SELECT TTL(v) FROM %s"));

        // Insert row without TTL
        execute("INSERT INTO %s (pk, ck, v) VALUES (1, 2, 3) USING TIMESTAMP ?", TIMESTAMP_1);
        assertWritetimeAndTTL("v", TIMESTAMP_1, NO_TTL);

        // Update the row with TTL and a new timestamp
        execute("UPDATE %s USING TIMESTAMP ? AND TTL ? SET v=8 WHERE pk=1 AND ck=2", TIMESTAMP_2, TTL_1);
        assertWritetimeAndTTL("v", TIMESTAMP_2, TTL_1);

        // Combine with other columns
        assertRows("SELECT pk, WRITETIME(v) FROM %s", row(1, TIMESTAMP_2));
        assertRows("SELECT WRITETIME(v), pk FROM %s", row(TIMESTAMP_2, 1));
        assertRows("SELECT pk, WRITETIME(v), v, ck FROM %s", row(1, TIMESTAMP_2, 8, 2));
    }

    @Test
    public void testList() throws Throwable
    {
        createTable("CREATE TABLE %s (k int PRIMARY KEY, l list<int>)");
        assertInvalidMultiCellSelection("l", true);
    }

    @Test
    public void testFrozenList() throws Throwable
    {
        createTable("CREATE TABLE %s (k int PRIMARY KEY, v frozen<list<int>>)");

        // Null column
        execute("INSERT INTO %s (k) VALUES (1) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("v", NO_TIMESTAMP, NO_TTL);

        // Create empty
        execute("INSERT INTO %s (k, v) VALUES (1, []) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("v", TIMESTAMP_1, TTL_1);

        // Update with a single element without TTL
        execute("INSERT INTO %s (k, v) VALUES (1, [1]) USING TIMESTAMP ?", TIMESTAMP_1);
        assertWritetimeAndTTL("v", TIMESTAMP_1, NO_TTL);

        // Add a new element to the list with a new timestamp and a TTL
        execute("INSERT INTO %s (k, v) VALUES (1, [1, 2, 3]) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_2, TTL_2);
        assertWritetimeAndTTL("v", TIMESTAMP_2, TTL_2);
    }

    @Test
    public void testSet() throws Throwable
    {
        createTable("CREATE TABLE %s (k int PRIMARY KEY, s set<int>)");
        assertInvalidMultiCellSelection("s", true);
    }

    @Test
    public void testFrozenSet() throws Throwable
    {
        createTable("CREATE TABLE %s (k int PRIMARY KEY, s frozen<set<int>>)");

        // Null column
        execute("INSERT INTO %s (k) VALUES (1) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("s", NO_TIMESTAMP, NO_TTL);

        // Create empty
        execute("INSERT INTO %s (k, s) VALUES (1, {}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("s", TIMESTAMP_1, TTL_1);

        // Update with a single element without TTL
        execute("INSERT INTO %s (k, s) VALUES (1, {1}) USING TIMESTAMP ?", TIMESTAMP_1);
        assertWritetimeAndTTL("s", TIMESTAMP_1, NO_TTL);

        // Add a new element to the set with a new timestamp and a TTL
        execute("INSERT INTO %s (k, s) VALUES (1, {1, 2}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_2, TTL_2);
        assertWritetimeAndTTL("s", TIMESTAMP_2, TTL_2);
    }

    @Test
    public void testMap() throws Throwable
    {
        createTable("CREATE TABLE %s (k int PRIMARY KEY, m map<int, int>)");
        assertInvalidMultiCellSelection("m", true);
    }

    @Test
    public void testFrozenMap() throws Throwable
    {
        createTable("CREATE TABLE %s (k int PRIMARY KEY, m frozen<map<int,int>>)");

        // Null column
        execute("INSERT INTO %s (k) VALUES (1) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("m", NO_TIMESTAMP, NO_TTL);

        // Create empty
        execute("INSERT INTO %s (k, m) VALUES (1, {}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("m", TIMESTAMP_1, TTL_1);

        // Create with a single element without TTL
        execute("INSERT INTO %s (k, m) VALUES (1, {1:10}) USING TIMESTAMP ?", TIMESTAMP_1);
        assertWritetimeAndTTL("m", TIMESTAMP_1, NO_TTL);

        // Add a new element to the map with a new timestamp and a TTL
        execute("INSERT INTO %s (k, m) VALUES (1, {1:10, 2:20}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_2, TTL_2);
        assertWritetimeAndTTL("m", TIMESTAMP_2, TTL_2);
    }

    @Test
    public void testUDT() throws Throwable
    {
        String type = createType("CREATE TYPE %s (f1 int, f2 int)");
        createTable("CREATE TABLE %s (k int PRIMARY KEY, t " + type + ')');
        assertInvalidMultiCellSelection("t", false);
    }

    @Test
    public void testFrozenUDT() throws Throwable
    {
        String type = createType("CREATE TYPE %s (f1 int, f2 int)");
        createTable("CREATE TABLE %s (k int PRIMARY KEY, t frozen<" + type + ">)");

        // Null column
        execute("INSERT INTO %s (k) VALUES (0) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("t", NO_TIMESTAMP, NO_TTL);

        // Both fields are empty
        execute("INSERT INTO %s (k, t) VALUES (0, {f1:null, f2:null}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("t", "k=0", TIMESTAMP_1, TTL_1);

        // Only the first field is set
        execute("INSERT INTO %s (k, t) VALUES (1, {f1:1, f2:null}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("t", "k=1", TIMESTAMP_1, TTL_1);

        // Only the second field is set
        execute("INSERT INTO %s (k, t) VALUES (2, {f1:null, f2:2}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("t", "k=2", TIMESTAMP_1, TTL_1);

        // Both fields are set
        execute("INSERT INTO %s (k, t) VALUES (3, {f1:1, f2:2}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("t", "k=3", TIMESTAMP_1, TTL_1);
    }

    @Test
    public void testFrozenNestedUDTs() throws Throwable
    {
        String nestedType = createType("CREATE TYPE %s (f1 int, f2 int)");
        String type = createType(format("CREATE TYPE %%s (f1 frozen<%s>, f2 frozen<%<s>)", nestedType));
        createTable("CREATE TABLE %s (k int PRIMARY KEY, t frozen<" + type + ">)");

        // Both fields are empty
        execute("INSERT INTO %s (k, t) VALUES (1, {f1:null, f2:null}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("t", "k=1", TIMESTAMP_1, TTL_1);

        // Only the first field is set, no nested field is set
        execute("INSERT INTO %s (k, t) VALUES (2, {f1:{f1:null,f2:null}, f2:null}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("t", "k=2", TIMESTAMP_1, TTL_1);

        // Only the first field is set, only the first nested field is set
        execute("INSERT INTO %s (k, t) VALUES (3, {f1:{f1:1,f2:null}, f2:null}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("t", "k=3", TIMESTAMP_1, TTL_1);

        // Only the first field is set, only the second nested field is set
        execute("INSERT INTO %s (k, t) VALUES (4, {f1:{f1:null,f2:2}, f2:null}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("t", "k=4", TIMESTAMP_1, TTL_1);

        // Only the first field is set, both nested field are set
        execute("INSERT INTO %s (k, t) VALUES (5, {f1:{f1:1,f2:2}, f2:null}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("t", "k=5", TIMESTAMP_1, TTL_1);

        // Only the second field is set, no nested field is set
        execute("INSERT INTO %s (k, t) VALUES (6, {f1:null, f2:{f1:null,f2:null}}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("t", "k=6", TIMESTAMP_1, TTL_1);

        // Only the second field is set, only the first nested field is set
        execute("INSERT INTO %s (k, t) VALUES (7, {f1:null, f2:{f1:1,f2:null}}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("t", "k=7", TIMESTAMP_1, TTL_1);

        // Only the second field is set, only the second nested field is set
        execute("INSERT INTO %s (k, t) VALUES (8, {f1:null, f2:{f1:null,f2:2}}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("t", "k=8", TIMESTAMP_1, TTL_1);

        // Only the second field is set, both nested field are set
        execute("INSERT INTO %s (k, t) VALUES (9, {f1:null, f2:{f1:1,f2:2}}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("t", "k=9", TIMESTAMP_1, TTL_1);

        // Both fields are set, alternate fields are set
        execute("INSERT INTO %s (k, t) VALUES (10, {f1:{f1:1}, f2:{f2:2}}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("t", "k=10", TIMESTAMP_1, TTL_1);

        // Both fields are set, alternate fields are set
        execute("INSERT INTO %s (k, t) VALUES (11, {f1:{f2:2}, f2:{f1:1}}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("t", "k=11", TIMESTAMP_1, TTL_1);

        // Both fields are set, all fields are set
        execute("INSERT INTO %s (k, t) VALUES (12, {f1:{f1:1,f2:2},f2:{f1:1,f2:2}}) USING TIMESTAMP ? AND TTL ?", TIMESTAMP_1, TTL_1);
        assertWritetimeAndTTL("t", "k=12", TIMESTAMP_1, TTL_1);
    }

    @Test
    public void testFunctions() throws Throwable
    {
        createTable("CREATE TABLE %s (k int PRIMARY KEY, v int, s set<int>, fs frozen<set<int>>)");
        execute("INSERT INTO %s (k, v, s, fs) VALUES (0, 0, {1, 2, 3}, {1, 2, 3}) USING TIMESTAMP 1 AND TTL 1000");
        execute("INSERT INTO %s (k, v, s, fs) VALUES (1, 1, {10, 20, 30}, {10, 20, 30}) USING TIMESTAMP 10 AND TTL 1000");
        execute("UPDATE %s USING TIMESTAMP 2 AND TTL 2000 SET s = s + {2, 3} WHERE k = 0");
        execute("UPDATE %s USING TIMESTAMP 20 AND TTL 2000 SET s = s + {20, 30} WHERE k = 1");

        // Regular column
        assertRows("SELECT min(v) FROM %s", row(0));
        assertRows("SELECT max(v) FROM %s", row(1));
        assertRows("SELECT writetime(v) FROM %s", row(10L), row(1L));
        assertRows("SELECT min(writetime(v)) FROM %s", row(1L));
        assertRows("SELECT max(writetime(v)) FROM %s", row(10L));

        // Frozen collection
        // Note that currently the tested system functions (min and max) return collections in their serialized format,
        // this is something that we might want to improve in the future (CASSANDRA-17811).
        assertRows("SELECT min(fs) FROM %s", row(ListType.getInstance(Int32Type.instance, false).decompose(Arrays.asList(1, 2, 3))));
        assertRows("SELECT max(fs) FROM %s", row(ListType.getInstance(Int32Type.instance, false).decompose(Arrays.asList(10, 20, 30))));
        assertRows("SELECT writetime(fs) FROM %s", row(10L), row(1L));
        assertRows("SELECT min(writetime(fs)) FROM %s", row(1L));
        assertRows("SELECT max(writetime(fs)) FROM %s", row(10L));

        // Multi-cell collection
        // Note that currently the tested system functions (min and max) return collections in their serialized format,
        // this is something that we might want to improve in the future (CASSANDRA-17811).
        assertRows("SELECT min(s) FROM %s", row(ListType.getInstance(Int32Type.instance, false).decompose(Arrays.asList(1, 2, 3))));
        assertRows("SELECT max(s) FROM %s", row(ListType.getInstance(Int32Type.instance, false).decompose(Arrays.asList(10, 20, 30))));
    }

    private void assertRows(String query, Object[]... rows) throws Throwable
    {
        assertRows(execute(query), rows);
    }

    private void assertWritetimeAndTTL(String column, Long timestamp, Integer ttl) throws Throwable
    {
        assertWritetimeAndTTL(column, null, timestamp, ttl);
    }

    private void assertWritetimeAndTTL(String column, String where, Long timestamp, Integer ttl)
    throws Throwable
    {
        where = where == null ? "" : " WHERE " + where;

        // Verify write time
        String writetimeQuery = String.format("SELECT WRITETIME(%s) FROM %%s %s", column, where);
        assertRows(writetimeQuery, row(timestamp));

        // Verify ttl
        UntypedResultSet rs = execute(String.format("SELECT TTL(%s) FROM %%s %s", column, where));
        assertRowCount(rs, 1);
        UntypedResultSet.Row row = rs.one();
        String ttlColumn = String.format("ttl(%s)", column);
        if (ttl == null)
        {
            assertTTL(ttl, null);
        }
        else
        {
            assertTTL(ttl, row.getInt(ttlColumn));
        }
    }

    /**
     * Since the returned TTL is the remaining seconds since last update, it could be lower than the
     * specified TTL depending on the test execution time, se we allow up to one-minute difference
     */
    private void assertTTL(Integer expected, Integer actual)
    {
        if (expected == null)
        {
            assertNull(actual);
        }
        else
        {
            assertNotNull(actual);
            assertTrue(actual > expected - 60);
            assertTrue(actual <= expected);
        }
    }

    private void assertInvalidPrimaryKeySelection(String column) throws Throwable
    {
        assertInvalidThrowMessage("Cannot use selection function writeTime on PRIMARY KEY part " + column,
                                  InvalidRequestException.class,
                                  String.format("SELECT WRITETIME(%s) FROM %%s", column));
        assertInvalidThrowMessage("Cannot use selection function ttl on PRIMARY KEY part " + column,
                                  InvalidRequestException.class,
                                  String.format("SELECT TTL(%s) FROM %%s", column));
    }

    private void assertInvalidMultiCellSelection(String column, boolean isCollection) throws Throwable
    {
        String message = format("Cannot use selection function %%s on non-frozen %s %s",
                                isCollection ? "collection" : "UDT", column);
        assertInvalidThrowMessage(format(message, "writeTime"),
                                  InvalidRequestException.class,
                                  String.format("SELECT WRITETIME(%s) FROM %%s", column));
        assertInvalidThrowMessage(format(message, "ttl"),
                                  InvalidRequestException.class,
                                  String.format("SELECT TTL(%s) FROM %%s", column));
    }
}
